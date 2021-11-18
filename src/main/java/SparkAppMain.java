import org.apache.hadoop.conf.Configuration;
import org.apache.spark.sql.SparkSession;
import edu.umd.cloud9.collection.XmlInputFormat;



public class SparkAppMain {
    public static double termDocWeight(int termFrequencyInDoc, int totalTermsInDoc, int termFreqInCorpus, int totalDocs) {
        double tf = ((double) termFrequencyInDoc) / totalTermsInDoc;
        double docFreq = ((double) totalDocs) / termFreqInCorpus;
        double idf = Math.log(docFreq);
        return tf*idf;
    }

    public static void main(String[] args) {
        SparkSession sparkSession = SparkSession.builder().appName("JavaWikipediaExample").getOrCreate();
        String path = "hdfs:///user/hdoop/data/wikidump.xml";
        Configuration conf = new Configuration();
        conf.set(XmlInputFormat.START_TAG_KEY, "<page>");
        conf.set(XmlInputFormat.END_TAG_KEY, "</page>");
    }
}
