package aprioriDataMining;

import java.util.ArrayList;

import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.java.tuple.Tuple2;

public class MinSupportFilter 
implements FilterFunction<Tuple2<Integer, ArrayList<Integer>>> {
    private double minSupportValue;
    
    MinSupportFilter(double filterValue){
    	this.minSupportValue=filterValue;
    }

    private static final long serialVersionUID = 1L;
	@Override
	public boolean filter(Tuple2<Integer, ArrayList<Integer>> tup) throws Exception {

       return (tup.f0 >= minSupportValue);
	  }
}

