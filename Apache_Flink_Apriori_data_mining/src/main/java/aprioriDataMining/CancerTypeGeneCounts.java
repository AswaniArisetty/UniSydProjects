package aprioriDataMining;

/* import the Apache-flink libraries */

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
//import org.apache.flink.api.java.operators.DataSink;
//import org.apache.flink.api.java.operators.SortPartitionOperator;
//import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.operators.Order;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.api.java.utils.ParameterTool;
//import org.apache.flink.core.fs.FileSystem.WriteMode;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.FileSystem;

import aprioriDataMining.DiseaseLineSplitter;
import aprioriDataMining.FilterCancerPatients;
import aprioriDataMining.ValidExpressionFilter;


public class CancerTypeGeneCounts {
	public static void main(String[] args) throws Exception {
		final ParameterTool params = ParameterTool.fromArgs(args);
		final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
		env.getConfig().setGlobalJobParameters(params); 
		int geneId=params.getInt("geneId",42);
		String outFile=params.getRequired("outFile");
		Double minExpnVal=params.getDouble("MinExpnValue",1250000);
		DataSet<Tuple3<String,Integer, Float>> geneData = env.readCsvFile(params.getRequired("geneDataFile"))
				.includeFields("111")
				.ignoreFirstLine()
				.fieldDelimiter(",")
				.lineDelimiter("\n")
				.types(String.class,Integer.class, Float.class);
		
		DataSet<Tuple3<String,Integer, Float>> geneDataFiltered=geneData
						.filter(new ValidExpressionFilter(geneId,minExpnVal))
						//.filter(new ValidExpressionFilter(42,10000))
						;//.returns(new TupleTypeInfo(TypeInformation.of(String.class), TypeInformation.of(Integer.class),TypeInformation.of(Float.class)));
		
		DataSet<Tuple3<String,String,Float>> patientData = env.readCsvFile(params.getRequired("patientDataFile"))
				.includeFields("100011")
				.ignoreFirstLine()
				.fieldDelimiter(",")
				.lineDelimiter("\n")
				.types(String.class,String.class, Float.class)
				;
		
		DataSet<Tuple3<String,String,Float>> mappatientData=patientData.flatMap(new DiseaseLineSplitter());
		//.returns(new TupleTypeInfo(TypeInformation.of(String.class), TypeInformation.of(String.class),TypeInformation.of(Float.class)));		
		
		DataSet<Tuple3<String,String, Float>> cancerPatients=mappatientData
				.filter(new FilterCancerPatients())
				;//.returns(new TupleTypeInfo(TypeInformation.of(String.class), TypeInformation.of(String.class),TypeInformation.of(Float.class)));

		DataSet<Tuple2<String, String>> patientGeneDiseases = cancerPatients.join(geneDataFiltered)
				.where(0)
				.equalTo(0)
				.projectFirst(1)
				.projectSecond(0)
				//.map(tuple -> new Tuple2<String, Integer>(tuple.f0,1))
				;
		DataSet<Tuple2<String , Integer>> outputTask1 = patientGeneDiseases
				.map(tuple -> new Tuple2<String,Integer>(tuple.f0,1) )
				.returns(new TupleTypeInfo(TypeInformation.of(String.class), TypeInformation.of(Integer.class)))
				.groupBy(0)
				.sum(1)
				.partitionByRange(1)
				.setParallelism(1)
				.sortPartition(1, Order.DESCENDING)
				.sortPartition(0, Order.ASCENDING);;
			
		outputTask1.writeAsCsv(outFile,"\n","\t",FileSystem.WriteMode.OVERWRITE);
		env.execute("GeneData Task1");
	}
    
}