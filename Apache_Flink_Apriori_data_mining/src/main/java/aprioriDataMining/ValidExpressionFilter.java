package aprioriDataMining;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.java.tuple.Tuple3;



public final class ValidExpressionFilter 
		implements FilterFunction<Tuple3<String, Integer, Float>> {
	    private double expressionFilter;
	    private float geneId;
	    
	    ValidExpressionFilter(Integer geneId,double filterValue){
	    	this.expressionFilter=filterValue;
	    	this.geneId=geneId;
	    }
	    
	    ValidExpressionFilter(double filterValue){
	    	this.expressionFilter=filterValue;
	    }
	    
	    private static final long serialVersionUID = 1L;
		@Override
		public boolean filter(Tuple3<String, Integer, Float> tup) throws Exception {
			if (geneId != 0.0f) { 
			return (tup.f1==geneId &&  tup.f2 >= expressionFilter);
		    }
           return (tup.f2 >= expressionFilter);
		  }
}
