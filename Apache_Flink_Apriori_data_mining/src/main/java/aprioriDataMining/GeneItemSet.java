package aprioriDataMining;

import java.util.ArrayList;
//import org.apache.commons.lang3.builder.EqualsBuilder;
//import org.apache.commons.lang3.builder.HashCodeBuilder;


public class GeneItemSet {

	public ArrayList<Integer> items;
	private int numberOfTransactions;

	public GeneItemSet() {
		this.items = new ArrayList<>();
		this.numberOfTransactions = 0;
	}

	public GeneItemSet(Integer item) {
		this.items = new ArrayList<>();
		this.items.add(item);
		this.numberOfTransactions = 1;
	}

	public GeneItemSet(ArrayList<Integer> itemList) {
		this.items = itemList;
	}



	public void setNumberOfTransactions(int numberOfTransactions) {
		this.numberOfTransactions = numberOfTransactions;
	}



	public int getNumberOfTransactions() {
		return numberOfTransactions;
	}

}
