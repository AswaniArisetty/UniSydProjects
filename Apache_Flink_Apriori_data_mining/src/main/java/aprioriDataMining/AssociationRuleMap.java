package aprioriDataMining;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;

public class AssociationRuleMap extends RichMapFunction<Tuple2<Integer , ArrayList<Integer>> 
						,Tuple2<Integer , ArrayList<Integer>>> 
						implements 	FlatMapFunction<Tuple2<Integer , ArrayList<Integer>>
													,Tuple3<ArrayList<Integer>,ArrayList<Integer>,Double>>
						{
	private static final long serialVersionUID = 1L;
	
	private Collection<Tuple2<Integer,ArrayList<Integer>>> geneIdSupports;
	private Map<ArrayList<Integer>,Integer> geneMap = new HashMap<ArrayList<Integer>,Integer>();
	private Float confidence;
	
	 AssociationRuleMap(Float confvalue){
		this.confidence=confvalue;
	}
	
	 AssociationRuleMap(){
	}
	
	@Override
	public void open(Configuration parameters) throws Exception {
		this.geneIdSupports = getRuntimeContext().getBroadcastVariable("freqItemsets");
		for (Tuple2<Integer,ArrayList<Integer>> i:this.geneIdSupports) this.geneMap.put(i.f1,i.f0);
	}
	
	public Tuple2<Integer , ArrayList<Integer>> map(Tuple2<Integer , ArrayList<Integer>> arg0) 
			throws Exception {
		return arg0;

}
	public void flatMap(Tuple2<Integer , ArrayList<Integer>> arg0
			, Collector<Tuple3<ArrayList<Integer>,ArrayList<Integer>,Double>> out) {
		Integer itemSetSupport=arg0.f0;
		//Map<Integer,Integer> map = new HashMap<Integer,Integer>();
		ArrayList<Integer> itemset=arg0.f1;
		if (itemset.size()>1){
/*		for (Integer item:itemset){
			ArrayList<Integer> itemset_sub=new ArrayList<Integer>();
			itemset_sub.addAll(itemset);
			Integer itemsupport=geneMap.get(item);
			Double itemconf=((double) itemSetSupport/itemsupport);
			if (itemconf > this.confidence){
				itemset_sub.remove(new Integer(item));
				out.collect(new Tuple3<Integer,ArrayList<Integer>,Double>(item,itemset_sub,itemconf));
			}
			
		}*/
		  for (int i=0;i<itemset.size();i++){
			  for (int j=i;j<=itemset.size();j++){
				  ArrayList<Integer> itemset_sub=new ArrayList<Integer>();
				  itemset_sub.addAll(itemset);
				  //int k;
				  //if (i==j && j!=itemset.size()-1){
			//		  k=j+1;
			//	  }
			//	  else {
			//		  k=j;
			//	  }
				  ArrayList<Integer> item;
				  if (i==j){
					   item = new ArrayList<Integer>(itemset.get(j));
				  }
				  else{
					  item = new ArrayList<Integer>(itemset.subList(i,j));
				  }
			if (item.size()!=itemset_sub.size()){
				  Integer itemsupport=geneMap.get(item);
				  if (itemsupport != null){
				  Double itemconf=((double) itemSetSupport/itemsupport);
					if (itemconf >= this.confidence){
						for (Integer b:item){
						itemset_sub.remove(b);
						}
						
						out.collect(new Tuple3<ArrayList<Integer>,ArrayList<Integer>,Double>(item,itemset_sub,itemconf));
					}
				  }
			}
		   }
			  
		 }
		}
}
}