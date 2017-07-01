package aprioriDataMining;


import java.util.ArrayList;
import java.util.Collections;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.tuple.Tuple2;

public class GeneItemsKeySelector implements KeySelector<Tuple2<Integer,ArrayList<Integer>>, String> {
	private static final long serialVersionUID = 1L;
	@Override
	public String getKey(Tuple2<Integer,ArrayList<Integer>> arg0) throws Exception {
		String key = null;
		//ArrayList<Integer> items = arg0.items;
		ArrayList<Integer>items=arg0.f1;
		Collections.sort(items);
		for (Integer item : items) {
			key += item.toString();
		}
		return key;
	}
}