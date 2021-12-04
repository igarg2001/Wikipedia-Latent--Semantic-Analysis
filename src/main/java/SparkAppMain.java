import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.IDF;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.ling.*;
import org.apache.spark.sql.types.*;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;


public class SparkAppMain {
    public static StanfordCoreNLP createNLPPipeline(){
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
        return new StanfordCoreNLP(props);
    }

    public static boolean isOnlyLetters(String str){
        for(char ch: str.toCharArray()){
            if(!Character.isLetter(ch))return false;
        }
        return true;
    }

    public static ArrayList<String> readFileIntoList(String file) {
        ArrayList<String> lines = new ArrayList<String>();
        try {
            lines = (ArrayList<String>) Files.readAllLines(Paths.get(file), StandardCharsets.UTF_8);
        } catch (IOException e) { // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return lines;
    }

    public static ArrayList<String> plainTextToLemmas(String text, ArrayList<String> StopWords, StanfordCoreNLP pipeline){
        ArrayList<String> lemmas = new ArrayList<String>();
        CoreDocument document = pipeline.processToCoreDocument(text);
        for(CoreLabel tok: document.tokens()){
            if(tok.lemma().length() > 2 && isOnlyLetters(tok.lemma()) && !StopWords.contains((tok.lemma())))
                lemmas.add(tok.lemma().toLowerCase());
        }
        return lemmas;
    }



    public static ArrayList<ArrayList<String>> topTermsInTopTopics(SingularValueDecomposition<RowMatrix, Matrix> svd, String[] termIds, int numConcepts, int numTerms){
        Matrix v = svd.V();
        ArrayList<ArrayList<String>> topTerms = new ArrayList<ArrayList<String>>();
        double[] arr = v.toArray();
        for(int i=0; i<numConcepts; i++){
            int offset = i * v.numRows();
            int len = offset + v.numRows();
            if(len > arr.length)len = arr.length;
            ArrayList<ArrayList<Double>> termWeights = new ArrayList<ArrayList<Double>>();
            for(int j=offset; j < len; j++){
                int index = j - offset;
                termWeights.add(new ArrayList<Double>(Arrays.asList(arr[j], (double)index)));
            }

            Collections.sort(termWeights, new Comparator<ArrayList<Double>>() {
                @Override
                public int compare(ArrayList<Double> t1, ArrayList<Double> t2) {
                    return t2.get(0).compareTo(t1.get(0));
                }
            });
            ArrayList<String> topTermslist = new ArrayList<String>();
            for(int j=0; j<numTerms; j++)
                try{
                    topTermslist.add(termIds[termWeights.get(j).get(1).intValue()]);
                }
                catch (Exception e){
                    continue;
                }
            topTerms.add(topTermslist);
        }
        return topTerms;
    }

    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder().appName("JavaWikipediaExample").getOrCreate();
        SparkContext sparkContext = sparkSession.sparkContext();
        JavaSparkContext sc = new JavaSparkContext(sparkContext);
        String path = "hdfs://localhost:9000/user/hdoop/data/enwiki.xml";
        Dataset<Row> newsgroupDF =
                sparkSession.read()
                        .format("csv")
                        .option("header", true)
                        .option("inferSchema", true)
                        .load("/home/xd101/20_newsgroup_10percent.csv");

        newsgroupDF = newsgroupDF.filter("text is not null");
        newsgroupDF.show();
        StructField[] schema = newsgroupDF.schema().fields();

        DataType lemmmatizedStruct = (DataType) DataTypes.createArrayType(DataTypes.StringType);
        StructField lemmatizedField = new StructField("lemmas", lemmmatizedStruct, true, Metadata.empty());
        StructField textField = new StructField("text", DataTypes.StringType, true, Metadata.empty());

        StructType schemas = DataTypes.createStructType(new StructField[]{textField, lemmatizedField});

        JavaRDD<Row> newsgroupRDD = newsgroupDF.toJavaRDD().map(new Function<Row, Row>() {
            public Row call(Row row) throws Exception {
                if (row != null) {
                    StanfordCoreNLP pipeline = createNLPPipeline();
                    String text = row.getString(1);
                    //updated row by creating new Row
                    return RowFactory.create(text, plainTextToLemmas(text, readFileIntoList("/home/xd101/Wikipedia-Latent--Semantic-Analysis/stopwords.txt"), pipeline).toArray());
                }
                return null;
            }
        });
        newsgroupDF = sparkSession.createDataFrame(newsgroupRDD, schemas);
        CountVectorizerModel cvModel = new CountVectorizer()
                .setInputCol("lemmas")
                .setOutputCol("rawFeatures")
                .fit(newsgroupDF);

        Dataset<Row> featurizedDataDF = cvModel.transform(newsgroupDF);

        IDF idf = new IDF()
                .setInputCol("rawFeatures")
                .setOutputCol("TFIDFfeatures");
        IDFModel idfModel = idf.fit(featurizedDataDF);

        Dataset<Row> documentTermMatrix = idfModel.transform(featurizedDataDF);
        documentTermMatrix = documentTermMatrix.cache();
        documentTermMatrix.show();


        JavaRDD<Vector> vecRDD = documentTermMatrix.select("TFIDFfeatures").toJavaRDD().map(new Function<Row, Vector>() {
            public Vector call(Row row) throws Exception {
                return (Vector) Vectors.fromML((org.apache.spark.ml.linalg.Vector) row.<Vector>getAs("TFIDFfeatures"));
            }

        });
        vecRDD = vecRDD.cache();

        RowMatrix mat = new RowMatrix(vecRDD.rdd());

        int k = 50;

        SingularValueDecomposition<RowMatrix, Matrix> svd = mat.computeSVD(k, true, 1.0E-9d);
        ArrayList<ArrayList<String>> topTerms = topTermsInTopTopics(svd, cvModel.vocabulary(), 10, 10);
        JavaRDD<ArrayList<String>> topTermsRDD = sc.parallelize(topTerms);
        topTermsRDD.take(50).forEach(s -> System.out.println(s));

        sparkSession.close();
    }
}
