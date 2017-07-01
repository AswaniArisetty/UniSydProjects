package aprioriDataMining;
import java.util.ArrayList;
import org.apache.flink.api.common.functions.GroupReduceFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.util.Collector;
public class PatientGroupReduceFunction implements GroupReduceFunction<Tuple2<Integer, String>, Tuple2<Integer, ArrayList<String>>> {
	private static final long serialVersionUID = 1L;
	@Override
	public void reduce(Iterable<Tuple2<Integer, String>> arg0, Collector<Tuple2<Integer, ArrayList<String>>> arg1)
			throws Exception {
		ArrayList<String> items = new ArrayList<>();
		Integer tid = null;
		for (Tuple2<Integer, String> transaction : arg0) {
			items.add(transaction.f1);
			tid = transaction.f0;
		}
		arg1.collect(new Tuple2<Integer, ArrayList<String>>(tid, items));
	}
}