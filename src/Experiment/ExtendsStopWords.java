package Experiment;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeSet;

import EmbeddingDiver.WordEntry;

public class ExtendsStopWords {

	ArrayList<String> stopArr = new ArrayList<String>();
	
	HashSet<String> stopSet = new HashSet<String>();
	HashMap<String, float[]> wordMap = new HashMap<String, float[]>();
	
	String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/vectors.840B.300d.txt";
	
	//String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/glove.6B.300d.txt";
	
	public void loadStopWords()
	{
		File txt = new File("data/stopwords/stopwords_extend.txt");	
		InputStreamReader is;
		String sw = null;
		try {
			is = new InputStreamReader(new FileInputStream(txt), "UTF-8");
			BufferedReader br = new BufferedReader(is);				
			while ((sw=br.readLine()) != null)  {
				stopArr.add(sw);   
				stopSet.add(sw);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	public void readVector()
	{
		try
		{
			BufferedReader br1 = new BufferedReader(new FileReader(vectorPath));
			String line = "";
			float vector = 0;
			while ((line = br1.readLine()) != null) {
			
				String word[] = line.split(" ");
				
				String word1 = word[0];
				float []vec = new float[word.length-1];
				for(int i=1; i<word.length; i++)
				{
					vector = Float.parseFloat(word[i]);///(word.length-1);
					vec[i-1] = vector;
				}	
				wordMap.put(word1, vec);
				
			}
			
			br1.close();
		}catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	
	public float wordSisCosine(float w1[], float w2[])
	{
		float dis = 0;
		
		float d1 = 0;
		
		float d2 = 0;
		for(int i=0; i<w1.length; i++)
		{
			//dis += w1[i]*w2[i];//*(w1[i]-w2[i]);
			d1 += w1[i]*w1[i];
			d2 += w2[i]*w2[i];
		}
		
		d1 = (float)Math.sqrt(d1);
		
		d2 = (float)Math.sqrt(d2);
		
		for(int i=0; i<w1.length; i++)
		{
			//dis += w1[i]*w2[i];//*(w1[i]-w2[i]);
			w1[i] /= d1;
			w2[i] /= d2;
		}
		d1 = 0;
		d2 = 0;
		for(int i=0; i<w1.length; i++)
		{
			dis += w1[i]*w2[i];//*(w1[i]-w2[i]);
			d1 += w1[i]*w1[i];
			d2 += w2[i]*w2[i];
		}
		return dis/((float)Math.sqrt(d1)*(float)Math.sqrt(d2));
	}
	
	
	public ArrayList<String> sisCloseWord(String queryWord, float thesh) {

		float[] center = wordMap.get(queryWord);
		ArrayList<String> closeWords = new ArrayList<String>();
		if (center == null) {
			return closeWords;
		}

		//int resultSize = wordMap.size() < topNSize ? wordMap.size() : topNSize;
		//TreeSet<WordEntry> result = new TreeSet<WordEntry>();

		//double min = Float.MIN_VALUE;
		for (Map.Entry<String, float[]> entry : wordMap.entrySet()) {
			float[] vector = entry.getValue();
			String word = entry.getKey();
			
			
			String token = word.toLowerCase();
   		 // String token = word;
	    	  	//Matcher m = p.matcher(token); // only save these strings only contains characters
	    	  //	if( !m.find()  && token.length()>2 && token.length()<25 )
	    	if( Character.isLetter(token.charAt(0))  && token.length()>2 )
	    	{
				if(stopSet.contains(token))
					continue;
				float sis = wordSisCosine(center,vector);
	
				if (sis >= thesh) {
					closeWords.add(token);
				}
	    	}
		}
		//result.pollFirst();

		return closeWords;
	}

	
	public void extendStop(float thesh)
	{
		for(int i=0; i<stopArr.size(); i++)
		{
			
			ArrayList<String> closeWords = sisCloseWord(stopArr.get(i),thesh);
			
			if(closeWords.size()>0)
			{
				//System.out.println(stopArr.get(i)+ "---------------");
				
				//System.out.println(closeWords.toString());
				stopArr.addAll(closeWords);
				
				for(int j=0; j<closeWords.size(); j++)
					stopSet.add(closeWords.get(j));
			}
		}
	}
	
	public void saveStopWords() throws Exception
	{
		FileWriter fw = new FileWriter("data/stopwords/stopwords_extend0.75.txt");
		BufferedWriter bw = new BufferedWriter(fw);
		//String line = "";
		
		for(int i=0; i<stopArr.size(); i++)
		{
			bw.write(stopArr.get(i));
			bw.newLine();
		}
		bw.close();
		fw.close();
	}
	
	public void mainExtend() throws Exception 
	{
		loadStopWords();
		
		readVector();
		int originalLen = stopArr.size();
		System.out.println(originalLen);
		extendStop(0.75f);
		
		int curLen = stopArr.size();
		System.out.println(curLen);
		
		saveStopWords();
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		try
		{
			ExtendsStopWords es = new ExtendsStopWords();
			es.mainExtend();
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		
	}

}
