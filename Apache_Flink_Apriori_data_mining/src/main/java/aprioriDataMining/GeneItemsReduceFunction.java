package aprioriDataMining;
import java.util.ArrayList;

import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.tuple.Tuple2;


public class GeneItemsReduceFunction implements ReduceFunction<Tuple2<Integer , ArrayList<Integer>>> {
	private static final long serialVersionUID = 1L;
	/**
	 * Returns a new ItemSet instance with 'frequency' as the sum of
	 * the two input ItemSets
	 *
	 */
	@Override
	public Tuple2<Integer , ArrayList<Integer>> reduce(Tuple2<Integer , ArrayList<Integer>> arg0, Tuple2<Integer , ArrayList<Integer>> arg1) throws Exception {
		Tuple2<Integer , ArrayList<Integer>> item =arg0;//= new Tuple2<Integer,ArrayList<Integer>>(arg0);
		item.f0=arg0.f0+arg1.f0;
		item.f1=arg0.f1;

		return item;
	}
}