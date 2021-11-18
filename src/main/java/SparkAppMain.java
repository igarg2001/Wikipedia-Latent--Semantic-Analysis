import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.*;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import edu.stanford.nlp.pipeline.*;
import edu.umd.cloud9.collection.XMLInputFormat;
import edu.umd.cloud9.collection.wikipedia.language.*;
import edu.umd.cloud9.collection.wikipedia.*;
import scala.Tuple2;



import java.util.ArrayList;
import org.apache.spark.api.java.Optional;

//class WikiFeature {
//    String title;
//    String content;
//    boolean isEmpty;
//
//    WikiFeature(){
//        isEmpty = true;
//        title = null;
//        content = null;
//    }
//
//    WikiFeature(String title, String content){
//        this.title = title;
//        this.content = content;
//        isEmpty = false;
//    }
//}

public class SparkAppMain {
    public static Optional<Tuple<String, String>> wikiXmlToPlainText(String xml) {
        EnglishWikipediaPage page = new EnglishWikipediaPage();
        WikipediaPage.readPage(page, xml);
        if(page.isEmpty()) return Optional.empty();
        else return Optional.of(new Tuple(page.getTitle(), page.getContent()));
    }

    public static double termDocWeight(int termFrequencyInDoc, int totalTermsInDoc, int termFreqInCorpus, int totalDocs) {
        double tf = ((double) termFrequencyInDoc) / totalTermsInDoc;
        double docFreq = ((double) totalDocs) / termFreqInCorpus;
        double idf = Math.log(docFreq);
        return tf*idf;
    }

    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder().appName("JavaWikipediaExample").getOrCreate();
        SparkContext sparkContext = sparkSession.sparkContext();
        String path = "hdfs:///user/hdoop/data/wikidump.xml";
        Configuration conf = new Configuration();
        conf.set(XMLInputFormat.START_TAG_KEY, "<page>");
        conf.set(XMLInputFormat.END_TAG_KEY, "</page>");
        JavaRDD<Tuple2<LongWritable, Text>> rdd = sparkContext.newAPIHadoopFile(path, XMLInputFormat.class, LongWritable.class, Text.class, conf).toJavaRDD();
        JavaRDD<String> rawXMLs = rdd.map(p -> p._2.toString());
    }
}
