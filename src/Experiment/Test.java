package Experiment;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.TreeSet;
import java.util.Vector;

import EmbeddingDiver.WordEntry;

public class Test {

	 static Random _rnd = new Random(1);
	
	public double pmi(int num1, int num2, int com)
	{
		int d = 2000;
		double com1 = (double)com/d;
		
		double num1P = (double)num1/d;
		double num2P = (double)num2/d;
		
		
		return Math.log(com1/(num1P*num2P));
	}
	
	public double docSis(ArrayList<Double> d1, ArrayList<Double> d2, ArrayList<Double> dist)
	{
		double sis = 0.0;
		
		ArrayList<Double> doc1 = new ArrayList<Double> ();
		doc1.addAll(d1);
		
		ArrayList<Double> doc2 = new ArrayList<Double> ();
		doc2.addAll(d2);
		
		ArrayList<Integer> indArr = new ArrayList<Integer>();
		
		for(int i=0; i<dist.size(); i++)
		{
			indArr.add(i);
		}
		
		for(int i=0; i<dist.size(); i++)
		{
			double cur = dist.get(i);
			
			int curInd = indArr.get(i);
			
			double minV = dist.get(i);
			int minInd = i;
			int ind = -1;
			for(int j=i+1; j<dist.size(); j++)
			{
				if(dist.get(j)<minV)
				{
					minV = dist.get(j);
					minInd = j;
					ind = indArr.get(j);
				}
			}
			if(i!=minInd)
			{
				
				dist.set(i, minV);
				dist.set(minInd, cur);
				
				indArr.set(i, ind);
				indArr.set(minInd, curInd);
			
			}
			//indArr.add(minInd);
		}
		
		for(int i=0; i<dist.size(); i++)
		{
			int index = indArr.get(i);
			
			int doc1Ind = index/doc2.size();
			int doc2Ind = index%doc2.size();
			
			double wei1 = doc1.get(doc1Ind);
			double wei2 = doc2.get(doc2Ind);
			
			if(wei1<1e-5 || wei2<1e-5)
				continue;
			
			double minWei = 0.0;
			if(wei1>wei2)
				minWei = wei2;
			else
				minWei = wei1;
			
			sis += minWei*dist.get(i);
			doc1.set(doc1Ind, wei1-minWei);
			doc2.set(doc2Ind, wei2-minWei);	
		}
		
		return sis;
	}
	
	public void test2(int num)
	{
		num = 5;
	}
	
	public void test()
	{
		for(int i=0; i<10; i++)
			System.out.print(_rnd.nextInt(20) + " ");
		
		//_rnd = null;
		System.out.println();
		//_rnd = new Random(1);
		for(int i=0; i<20; i++)
			System.out.print(_rnd.nextInt(30) + " ");
		
		int num = 0;
		
		test2(num);
		
		System.out.println(num);
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		Test t = new Test();
		
		t.test();
		
		
	}

}
