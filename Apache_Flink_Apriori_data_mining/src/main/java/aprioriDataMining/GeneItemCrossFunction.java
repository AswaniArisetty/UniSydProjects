package aprioriDataMining;

import java.util.ArrayList;
import org.apache.flink.api.common.functions.CrossFunction;
import org.apache.flink.api.java.tuple.Tuple2;
public class GeneItemCrossFunction implements CrossFunction<Tuple2<Integer,ArrayList<Integer>>
, Tuple2<Integer,ArrayList<Integer>>, Tuple2<Integer,ArrayList<Integer>>> {
	private static final long serialVersionUID = 1L;
	@Override
	public Tuple2<Integer,ArrayList<Integer>> cross(Tuple2<Integer , ArrayList<Integer>> arg0
			, Tuple2<Integer , ArrayList<Integer>> arg1) throws Exception {
		// create a new ArrayList of items
		ArrayList<Integer> items = arg0.f1;
		//ArrayList<String> patients=arg0.f2;
		// only add new items
		for (Integer item : arg1.f1) {
			if (!items.contains(item)) {
				items.add(item);
			}
		}
		// create a new ItemSet
		Tuple2<Integer,ArrayList<Integer>> newItemSet = new Tuple2<Integer,ArrayList<Integer>>(arg0.f0,items);
		// set a temporary number of transactions
		//newItemSet.setNumberOfTransactions(arg0.getNumberOfTransactions());
		return newItemSet;
	}
}