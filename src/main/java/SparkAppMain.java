import static org.apache.spark.sql.functions.size;
import static org.apache.spark.sql.functions.udf;

import java.lang.Exception;
import java.lang.String;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.UserDefinedFunction;

public class SparkAppMain {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().appName("").getOrCreate();
        SparkContext sparkContext = spark.sparkContext();
        JavaSparkContext javaSparkContext = new JavaSparkContext(sparkContext);
        Dataset<Row> dataset = spark.read().format("csv").option("header", true).option("inferSchema", true).load("/home/xd101/20_newsgroup_10percent.csv");
        dataset = dataset.filter("text is not null");
        UserDefinedFunction lemmatizer = udf((java.lang.String x) -> {
            java.util.Properties props = new java.util.Properties();
            props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
            edu.stanford.nlp.pipeline.StanfordCoreNLP pipeline = new edu.stanford.nlp.pipeline.StanfordCoreNLP(props);
            java.util.ArrayList<java.lang.String> stopWords = new java.util.ArrayList<java.lang.String>();
            try {
                stopWords = (java.util.ArrayList<java.lang.String>) java.nio.file.Files.readAllLines(java.nio.file.Paths.get("/home/xd101/Wikipedia-Latent--Semantic-Analysis/stopwords.txt"), java.nio.charset.StandardCharsets.UTF_8);
            } catch (java.io.IOException e) {
                e.printStackTrace();
            }
            java.util.ArrayList<java.lang.String> lemmas = new java.util.ArrayList<java.lang.String>();
            edu.stanford.nlp.pipeline.CoreDocument document = pipeline.processToCoreDocument(x);
            for(edu.stanford.nlp.ling.CoreLabel tok: document.tokens()) {
                if(tok.lemma().length() > 2 && !stopWords.contains(tok.lemma())) {
                    boolean flag = false;
                    for(char ch: tok.lemma().toCharArray()) {
                        if(!java.lang.Character.isLetter(ch)) {
                            flag = true;
                            break;
                        }
                    }
                    if(flag) {
                        break;
                    }
                    lemmas.add(tok.lemma().toLowerCase());
                }
            }
            return lemmas;
        }, org.apache.spark.sql.types.DataTypes.createArrayType(org.apache.spark.sql.types.DataTypes.StringType));
        dataset = dataset.withColumn("lemmas", lemmatizer.apply(dataset.col("text")));
        dataset = dataset.filter(size(dataset.col("lemmas")).$greater(0));
        dataset.show();
        CountVectorizerModel model = new CountVectorizer().setInputCol("lemmas").setOutputCol("rawFeatures").fit(dataset);
        dataset = model.transform(dataset);
        dataset.show();
        IDFModel idfModel = new IDF().setInputCol("rawFeatures").setOutputCol("TFIDFfeatures").fit(dataset);
        dataset = idfModel.transform(dataset);
        dataset.show();
        dataset.cache();
        JavaRDD<Vector> vecRDD = dataset.select("TFIDFfeatures").toJavaRDD().map((org.apache.spark.sql.Row row) -> {
            return org.apache.spark.mllib.linalg.Vectors.fromML((org.apache.spark.ml.linalg.Vector)row.<org.apache.spark.mllib.linalg.Vector>getAs("TFIDFfeatures"));
        });
        vecRDD = vecRDD.cache();
        RowMatrix mat = new RowMatrix(vecRDD.rdd());
        SingularValueDecomposition<RowMatrix, Matrix> svd = mat.computeSVD(10, true, 1.0E-9d);
        Matrix v = svd.V();
        String[] termIds = model.vocabulary();
        ArrayList<ArrayList<String>> topTerms = new ArrayList<ArrayList<String>>();
        double[] arr = v.toArray();
        for(int i=0; i<10; i++) {
            int offset = i * v.numRows();
            int len = offset + v.numRows();
            if(len > arr.length) {
                len = arr.length;
            }
            ArrayList<ArrayList<Double>> termWeights = new ArrayList<ArrayList<Double>>();
            for(int j=offset; j<len; j++) {
                int index = j - offset;
                termWeights.add(new ArrayList<Double>(Arrays.asList(arr[j], (double)index)));
            }
            Collections.sort(termWeights, (ArrayList<Double> t1, ArrayList<Double> t2) -> t2.get(0).compareTo(t1.get(0)));
            ArrayList<String> topTermslist = new ArrayList<String>();
            for(int j=0; j<10; j++) {
                try {
                    topTermslist.add(termIds[termWeights.get(j).get(1).intValue()]);
                } catch(Exception e) {
                    continue;
                }
            }
            topTerms.add(topTermslist);
        }
        JavaRDD<ArrayList<String>> topTermsRDD = javaSparkContext.parallelize(topTerms);
//        topTermsRDD.saveAsTextFile("/home/xd101/topterms");
        topTermsRDD.foreach(x -> System.out.println(x));
        spark.stop();
    }
}