package aprioriDataMining;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.java.tuple.Tuple3;

public final class FilterCancerPatients 
			implements FilterFunction<Tuple3<String, String, Float>> {

	private static final long serialVersionUID = 1L;
	private String[] cancers={"breast-cancer","prostate-cancer","pancreatic-cancer","leukemia","lymphoma"};
	
	FilterCancerPatients(String[] cancerTypes){
    	this.cancers=cancerTypes;    	
    }

    FilterCancerPatients(){
    }
    
    @Override
	public boolean filter(Tuple3<String, String, Float> tup) throws Exception {
    	String[] diseases=tup.f1.split(" ");
    	for (String cancer:cancers){
    		for (String disease:diseases){
    	    	if (cancer.equals(disease)) 
    		    {
    		      return true;	
    		    }
    	    }
    	}
		return false;
	 }
 }