package aprioriDataMining;
import java.util.ArrayList;
import java.util.Collection;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
public class GeneSetCalculateFrequency extends RichMapFunction<Tuple2<Integer , ArrayList<Integer>> 
										,Tuple2<Integer , ArrayList<Integer>>> {
	private static final long serialVersionUID = 1L;
	private Collection<Tuple2<String, ArrayList<Integer>>> patients;
	@Override
	public void open(Configuration parameters) throws Exception {
		this.patients = getRuntimeContext().getBroadcastVariable("patients");
	}
	@Override
	public Tuple2<Integer , ArrayList<Integer>> map(Tuple2<Integer , ArrayList<Integer>> arg0) throws Exception {
		Tuple2<Integer , ArrayList<Integer>> out= arg0;
		int numberOfPatients = 0;
		for (Tuple2<String, ArrayList<Integer>> patient : this.patients) {
			if (patient.f1.containsAll(arg0.f1)) {
				numberOfPatients++;
			}
		}
		out.f0=numberOfPatients;
		return out;
	}
}