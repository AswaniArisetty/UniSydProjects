package aprioriDataMining;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.util.Collector;

public class DiseaseLineSplitter implements FlatMapFunction<Tuple3<String,String,Float>, Tuple3<String,String,Float>> {
        @Override
        public void flatMap(Tuple3<String,String,Float> line, Collector<Tuple3<String,String,Float>> out) {
        	String[] diseases=line.f1.split(" ");
			for (String disease:diseases){
				out.collect(new Tuple3<String, String,Float>(line.f0,disease.trim(),line.f2));
			}
        }
    }

