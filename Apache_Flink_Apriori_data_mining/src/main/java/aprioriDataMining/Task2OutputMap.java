package aprioriDataMining;

import java.util.ArrayList;

import java.util.Collections;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;

public class Task2OutputMap extends RichMapFunction<Tuple2<Integer , ArrayList<Integer>> 
										,Tuple2<Integer , String>> {
	private static final long serialVersionUID = 1L;
	public Tuple2<Integer , String> map(Tuple2<Integer , ArrayList<Integer>> arg0) throws Exception {
	ArrayList<Integer> n1=arg0.f1;
	Integer c1=arg0.f0;
	Collections.sort(n1);
	String t1=n1.toString().replaceAll(",", "\t").replace("[","").replace("]","");
	return new Tuple2<Integer,String>(c1,t1);
	}
	

}
