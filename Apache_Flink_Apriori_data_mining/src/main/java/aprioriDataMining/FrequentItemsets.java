package aprioriDataMining;

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.flink.api.common.operators.Order;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.typeutils.TupleTypeInfo;
import org.apache.flink.api.java.utils.ParameterTool;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.FileSystem;

import aprioriDataMining.FilterCancerPatients;
import aprioriDataMining.Task2OutputMap;
import aprioriDataMining.ValidExpressionFilter;

import org.apache.flink.api.java.operators.IterativeDataSet;

public class FrequentItemsets {
	public static void main(String[] args) throws Exception {
		final ParameterTool params = ParameterTool.fromArgs(args);
		final ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
		env.getConfig().setGlobalJobParameters(params); 
		String outFile_task2=params.getRequired("outFileTask2");
		String outFile_task3=params.getRequired("outFileTask3");
		Double minExpnVal=params.getDouble("MinExpnValue",1250000);
		Float minConf=params.getFloat("MinConfidence",0.6f);
		Float minSupportPerc=params.getFloat("MinSupportPerc", 0.3f);
		Integer maxItemsetSize= params.getInt("MaxItemsetSize",10);
		DataSet<Tuple3<String,Integer, Float>> geneData = env.readCsvFile(params.getRequired("geneDataFile"))
				.includeFields("111")
				.ignoreFirstLine()
				.fieldDelimiter(",")
				.lineDelimiter("\n")
				.types(String.class,Integer.class, Float.class);
		
		DataSet<Tuple3<String,Integer, Float>> geneDataFiltered=geneData
						.filter(new ValidExpressionFilter(minExpnVal))
						;
		
		DataSet<Tuple3<String,String,Float>> patientData = env.readCsvFile(params.getRequired("patientDataFile"))
				.includeFields("100011")
				.ignoreFirstLine()
				.fieldDelimiter(",")
				.lineDelimiter("\n")
				.types(String.class,String.class, Float.class)
				;
		
		DataSet<Tuple3<String,String, Float>> cancerPatients=patientData
				.filter(new FilterCancerPatients())
				;
		
		long totalCancerPatients = cancerPatients.distinct(0).count();
		long minSupportCount=(long)(totalCancerPatients*minSupportPerc);
		
		DataSet<Tuple2<Integer, String>> geneIdpatient = cancerPatients.join(geneDataFiltered)
				.where(0)
				.equalTo(0)
				//.projectFirst(1)
				.projectSecond(1,0);
	
		
		
		DataSet<Tuple2<String, ArrayList<Integer>>> patients = geneIdpatient
				.map((tuple)->new Tuple2<String,ArrayList<Integer>>(tuple.f1,new ArrayList<Integer>(Arrays.asList(tuple.f0))))
				.returns(new TupleTypeInfo(TypeInformation.of(String.class), TypeInformation.of(ArrayList.class)))
				.groupBy(0)
				.reduce(new PatientReduceFunction());
		

		DataSet<Tuple2<Integer, Integer>> geneIdSupports=geneIdpatient
							.map((tuple)-> new Tuple2<Integer,Integer>(tuple.f0,1))
							.returns(new TupleTypeInfo(TypeInformation.of(Integer.class),TypeInformation.of(Integer.class)))
							.groupBy(0)
							.sum(1)
							;
		
		DataSet<Tuple2<Integer , ArrayList<Integer>>> c1 = geneIdpatient
				.map(tuple -> new Tuple2<Integer,ArrayList<Integer>>(1,new ArrayList<Integer>(Arrays.asList(tuple.f0))))
				.returns(new TupleTypeInfo(TypeInformation.of(Integer.class),TypeInformation.of(ArrayList.class)))
				.groupBy(new GeneItemsKeySelector())
				.reduce(new GeneItemsReduceFunction())
				.filter(new MinSupportFilter(minSupportCount))
				//.partitionByRange(new GeneItemsKeySelector())
				.sortPartition(0, Order.DESCENDING)
				//.sortPartition(1, Order.ASCENDING)
				;
	    
		IterativeDataSet<Tuple2<Integer , ArrayList<Integer>>> initial = c1.iterate(maxItemsetSize-1);
		//DataSet<Tuple2<Integer , ArrayList<Integer>>> initial = c1;
		
		DataSet<Tuple2<Integer , ArrayList<Integer>>> candidates=initial.cross(c1)
													.with(new GeneItemCrossFunction())
													.distinct(new GeneItemsKeySelector());

		DataSet<Tuple2<Integer , ArrayList<Integer>>> selected=candidates
				.map(new GeneSetCalculateFrequency()).withBroadcastSet(patients, "patients")
				.filter(new MinSupportFilter(minSupportCount))
				//.partitionByRange(0)//(new GeneItemsKeySelector())
				//.sortPartition(0, Order.DESCENDING)
				//.sortPartition(1, Order.ASCENDING);
				;
		
		DataSet <Tuple2<Integer,ArrayList<Integer>>> freqItemsets=initial.closeWith(selected, selected);
		
		DataSet<Tuple2<Integer , String>> output_task2 = freqItemsets
																	//.partitionByRange(0)
																	.map(new Task2OutputMap())
																	.sortPartition(0, Order.DESCENDING)
																	.sortPartition(1, Order.ASCENDING)
																	.setParallelism(1);
		
		
		
		output_task2.writeAsCsv(outFile_task2,"\n","\t",FileSystem.WriteMode.OVERWRITE);
		
		DataSet <Tuple3<String,String,Double>> output_task3 = freqItemsets
						.flatMap(new AssociationRuleMap(minConf))
						.withBroadcastSet(freqItemsets, "freqItemsets")
						.map((tuple)-> new Tuple3<String,String,Double>(
								tuple.f0.toString().replace(",","\t").replace("[","").replace("]","")
								,tuple.f1.toString().replace(",","\t").replace("[","(").replace("]",")")
								,Math.round(tuple.f2*1000D)/1000D)
						    )
						.returns(new TupleTypeInfo(TypeInformation.of(String.class)
								,TypeInformation.of(String.class)
								,TypeInformation.of(Double.class)))
						.sortPartition(2, Order.DESCENDING)
						.setParallelism(1)
						;
		

		output_task3.writeAsCsv(outFile_task3,"\n","\t",FileSystem.WriteMode.OVERWRITE);
	
		env.execute("GeneData Apriori and Association RulesS");
		//System.out.print(env.getExecutionPlan());

}
}
