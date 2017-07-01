package aprioriDataMining;
import java.util.ArrayList;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.tuple.Tuple2;

public class PatientReduceFunction implements ReduceFunction<Tuple2<String,ArrayList<Integer>>> {
	private static final long serialVersionUID = 1L;
	@Override
	public Tuple2<String,ArrayList<Integer>> reduce(Tuple2<String,ArrayList<Integer>> arg0, Tuple2<String,ArrayList<Integer>> arg1) throws Exception {
		Tuple2<String,ArrayList<Integer>> item=new Tuple2<String,ArrayList<Integer>>(arg0.f0,arg0.f1);
		item.f1.addAll(arg1.f1);
	
	
	return item;
}
}
