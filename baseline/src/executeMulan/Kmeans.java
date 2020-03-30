package executeMulan;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
public class Kmeans {
public static void main(String args[]) throws Exception
{
	//String input_file = "/home/s/Dropbox/ML/Project/data/yelp_train.arff";
	String input = "";
	String output = "";
	String input_file = input + "bibtex.arff";
	Instances data = new Instances(new BufferedReader(new FileReader(input_file)));
	int k_value = 4;
	String output0 = output+"bibtex0.arff";
	FileWriter fw = new FileWriter(output0);
	BufferedWriter bw = new BufferedWriter(fw);
	
	String output1 = output+ "bibtex1.arff";
	FileWriter fw1 = new FileWriter(output1);
	BufferedWriter bw1 = new BufferedWriter(fw1);
	
	String output2 = output+ "bibtex2.arff";
	FileWriter fw2 = new FileWriter(output2);
	BufferedWriter bw2 = new BufferedWriter(fw2);
	
	String output3 = output+ "bibtex3.arff";
	FileWriter fw3 = new FileWriter(output3);
	BufferedWriter bw3 = new BufferedWriter(fw3);
	
	/*
	String output4 = output+ "bibtex4.arff";
	FileWriter fw4 = new FileWriter(output4);
	BufferedWriter bw4 = new BufferedWriter(fw4);
	

	String output5 = output+ "bibtex5.arff";
	FileWriter fw5 = new FileWriter(output5);
	BufferedWriter bw5 = new BufferedWriter(fw5);
	
	String output6 = output+ "bibtex6.arff";
	FileWriter fw6 = new FileWriter(output6);
	BufferedWriter bw6 = new BufferedWriter(fw6);
	

	String output7 = output+ "bibtex7.arff";
	FileWriter fw7 = new FileWriter(output7);
	BufferedWriter bw7 = new BufferedWriter(fw7);
	
	*/
	
	
	
	SimpleKMeans kmeans = new SimpleKMeans();
	kmeans.setPreserveInstancesOrder(true);
	kmeans.setNumClusters(k_value);
	kmeans.buildClusterer(data);
	
	 int n1=0,i=0,n2=0;
	 int n[] = new int[4];
	 int k[] = kmeans.getAssignments();
	 
	 
	 for(int clusterNum : k)
	 {
		 //System.out.printf("Instance %d -> cluster %d \n", i, clusterNum);		 
		 if(clusterNum==0)
		 {
			 bw.write(data.instance(i).toString());
			 bw.newLine();
			 n[0]++;
		 }
		 if(clusterNum==1)
		 {
			 bw1.write(data.instance(i).toString());
		     bw1.newLine();
		     n[1]++;
		 }
		 if(clusterNum==2)
		 {
			 bw2.write(data.instance(i).toString());
		     bw2.newLine();
		     n[2]++;
		 }
		 if(clusterNum==3)
		 {
			 bw3.write(data.instance(i).toString());
		     bw3.newLine(); 
		     n[3]++;
		 }
		 /*
		 if(clusterNum==4)
		 {
			 bw4.write(data.instance(i).toString());
		     bw4.newLine(); 
		     n[4]++;
		 }
		 if(clusterNum==5)
		 {
			 bw5.write(data.instance(i).toString());
		     bw5.newLine(); 
		     n[5]++;
		 }
		 if(clusterNum==6)
		 {
			 bw6.write(data.instance(i).toString());
		     bw6.newLine(); 
		     n[6]++;
		 }
		 if(clusterNum==7)
		 {
			 bw7.write(data.instance(i).toString());
		     bw7.newLine(); 
		     n[7]++;
		 }*/
		 
		 
		 
		 
		 i++;
	 }
	 System.out.println(n[0] +"--"+  n[1]+ "--"+n[2] + "--" + n[3] );
	 
	 bw.close();
	 bw1.close();
	 bw2.close();
	 bw3.close();
	 /*
	 bw4.close();
	 bw5.close();
	 bw6.close();
	 bw7.close(); */
	 /*
	 while(n<k.length)
	 {
		// System.out.println(k[n] + "   " + n);
		 if(k[n]==0)
		 {
			 i++;
			bw.write(data.instance(n).toString());
			bw.newLine();
		 }
		 else if (k[n]==1)
		 {
			 j++;
			 bw1.write(data.instance(n).toString());
				bw1.newLine();
			// bw1.write(data.instance(n).toString());
			 //bw1.newLine();
		 }
		 else
		 {
			 System.out.println("error occurred");
		 }
		 n++;
	 }
	 //System.out.println(data1);
	 System.out.println(i);
	 bw.close();
	 bw1.close();
		*/
	}
}
