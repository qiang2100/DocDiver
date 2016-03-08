package Experiment;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;

import EmbeddingDiver.WordClusteringForDiver;

public class ExperiBasedOnLDAData {

	ArrayList<ArrayList<String>> topic2Arr = new ArrayList<ArrayList<String>>();
	
	HashMap<String, float[]> wordMap = new HashMap<String, float[]>();
	
	
	
	String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/glove.6B.300d.txt";
	
	public void readVector()
	{
		try
		{
			BufferedReader br1 = new BufferedReader(new FileReader(vectorPath));
			String line = "";
			//int num = 0;
			
			//FileWriter subfw = new FileWriter("C:/Users/qjp/Desktop/UMAB/Word2Vec/glove.6B.300d.word.txt");
			
			//BufferedWriter subbw = new BufferedWriter(subfw);
			
			float vector = 0;
			while ((line = br1.readLine()) != null) {
			
				String word[] = line.split(" ");
				
				//allWord.add(word[0]);
				String word1 = word[0];
				float []vec = new float[word.length-1];
				//double len = 0;
				for(int i=1; i<word.length; i++)
				{
					vector = Float.parseFloat(word[i]);///(word.length-1);
					
					//len += vector * vector;
					vec[i-1] = vector;
				}
				/*len = Math.sqrt(len);

				for (int j = 0; j < vec.length; j++) {
					vec[j] /= len;
				}*/
				
				wordMap.put(word1, vec);
				//System.out.println(word.length);
				//break;
				//subbw.write(word[0]);
				//subbw.newLine();
				//num++;
			}
			//System.out.println(word.length);
			//System.out.println(allWord[1000] + vect[1000][0] + " " + vect[1000][1]);
			//subbw.close();
			br1.close();
		}catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	
	public void readTopic(String path) throws Exception
	{
		 BufferedReader br1 = new BufferedReader(new FileReader(path));
		String line = "";
		while ((line = br1.readLine()) != null) 
		{
			String []words = line.split(" ");
			
			ArrayList<String> oneTopicArr = new ArrayList<String>();
			for(int i=0; i<words.length; i++)
			{
				oneTopicArr.add(words[i]);
			}
			topic2Arr.add(oneTopicArr);
		}
		br1.close();
	}
	
	//choose the "2*num" words from one topic, and choose the "num" words from two different topic,respectively. Then, compute the score of diversity. 
	public  void computArrForWord2Vec(WordClusteringForDiver ed, int num)
	{
		if(2*num>topic2Arr.get(0).size())
			return;
		
		int right = 0;
		   int total = 0;
		
			/*String s2 = "computer";
			float []v2 = ed.wordMap.get(s2);
			for(int ii=0; ii<v2.length; ii++)
				System.out.print(v2[ii]+ " ");
			System.out.println();*/
		   
		for(int i=0; i<topic2Arr.size(); i++)
		{
			ArrayList<String> oneTopic = new ArrayList<String>();
			oneTopic.addAll(topic2Arr.get(i).subList(0, 2*num));
				
			//double oneScore = ed.compuDiverBasedKmeans(oneTopic);
			double oneScore = ed.compuDiver(oneTopic);
			
			//System.out.println("OneScore:" + oneScore);
		////	System.out.println("------------------------------");
			//System.out.println("the " + i + " oneScore: " + oneScore);
			
			for(int j=0; j<topic2Arr.size(); j++)
			{
				if(i==j)
					continue;
				ArrayList<String> twoTopic = new ArrayList<String>();
				twoTopic.addAll(topic2Arr.get(i).subList(0, num));
				twoTopic.addAll(topic2Arr.get(j).subList(0, num));
				
				//double twoScore = ed.compuDiverBasedKmeans(twoTopic);
				double twoScore = ed.compuDiver(twoTopic);
				//System.out.println("twoScore:" + twoScore);
				/*float []v3 = ed.wordMap.get(s2);
				for(int ii=0; ii<v3.length; ii++)
					System.out.print(v3[ii]+ " ");
				System.out.println();*/
				//System.out.println("the " + j + " twoScore: " + twoScore);
				total++;
				if(twoScore>oneScore)
					right++;
				else
					System.out.println("i:" + i + ":score: "+ oneScore+ " j:"+ j + ":score"+twoScore);
				
			}
			
		}
		
		
		System.out.println("Accury: " + (double)right/total);
	}
	
	public ArrayList<Integer> computWordFre(ArrayList<String>doc, ArrayList<String> diff)
	{
		ArrayList<Integer> textWordFre = new ArrayList<Integer>();
		
		for(int i=0; i<doc.size(); i++)
		{
			String word = doc.get(i);
			if(diff.contains(word))
			{
				int ind = diff.indexOf(word);
				textWordFre.set(ind, textWordFre.get(ind)+1);
			}else
			{
				diff.add(word);
				textWordFre.add(1);
			}
		}
		
		return textWordFre;
	}
	
	
	public void printVector(int num) throws Exception
	{
		int sI = 0;
		int bJ = 4;
		
		ArrayList<String> oneTopic = new ArrayList<String>();
		oneTopic.addAll(topic2Arr.get(sI).subList(0, 2*num));
		
		ArrayList<String> diff = new ArrayList<String>();
		
		ArrayList<Integer> textWordFre = computWordFre(oneTopic, diff);
		
		
		FileWriter pFW = new FileWriter("wordForVST.txt");
		BufferedWriter wordBW = new BufferedWriter(pFW);
		
		FileWriter tFW = new FileWriter("textVectorST.txt");
		BufferedWriter textBW = new BufferedWriter(tFW);
		
		readVector();
		
		for(int i=0; i<diff.size(); i++)
		{
			if(wordMap.containsKey(diff.get(i)))
			{
				wordBW.write(diff.get(i));
				wordBW.newLine();
				
				float vec[] = wordMap.get(diff.get(i));
				
				for(int j=0; j<vec.length; j++)
				{
					textBW.write(String.valueOf(vec[j])+ " ");
				}
				textBW.newLine();
			}
			
			
		}
		
		wordBW.close();
		textBW.close();
		
		
		ArrayList<String> twoTopic = new ArrayList<String>();
		twoTopic.addAll(topic2Arr.get(sI).subList(0, num));
		twoTopic.addAll(topic2Arr.get(bJ).subList(0, num));
		
		diff = new ArrayList<String>();
		
		textWordFre = computWordFre(twoTopic, diff);
		
		
		 pFW = new FileWriter("wordForVBT.txt");
		 wordBW = new BufferedWriter(pFW);
		
		 tFW = new FileWriter("textVectorBT.txt");
		 textBW = new BufferedWriter(tFW);
		
		
		
		for(int i=0; i<diff.size(); i++)
		{
			if(wordMap.containsKey(diff.get(i)))
			{
				wordBW.write(diff.get(i));
				wordBW.newLine();
				
				float vec[] = wordMap.get(diff.get(i));
				
				for(int j=0; j<vec.length; j++)
				{
					textBW.write(String.valueOf(vec[j])+ " ");
				}
				textBW.newLine();
			}
			
			
		}
		
		wordBW.close();
		textBW.close();
		
	}
	
	public  void computArrForWord2Vec2(WordClusteringForDiver ed, int num)
	{
		if(2*num>topic2Arr.get(0).size())
			return;
		
		
		
		
		for(int i=0; i<topic2Arr.size(); i++)
		{
			System.out.println("------------------------------");
			ArrayList<String> oneTopic = new ArrayList<String>();
			oneTopic.addAll(topic2Arr.get(i).subList(0, 2*num));
			System.out.println("Length: " + oneTopic.size());
			//double oneScore = ed.compuDiver(oneTopic);
			
			
			//System.out.println("the " + i + " oneScore: " + oneScore);
			
		}
	}
	
	public void mainForWord2Vec()
	{
		System.out.println("Embedding");
		WordClusteringForDiver ed = new WordClusteringForDiver();
		ed.readVector();
		computArrForWord2Vec(ed,100);
	}
	
	public void mainForWord2VecGroup()
	{
		WordClusteringForDiver ed = new WordClusteringForDiver();
		ed.readVector();
		//computArrForWord2Vec(ed,50);
		
		//readVector();
		
		for(double bigP = 0.25; bigP<=0.6; bigP += 0.05)
		{
			for(double smallP=0.15; (smallP<=0.4) && (smallP<=bigP-0.1); smallP += 0.05)
			{
				ed.initialPara(bigP,smallP);
				System.out.println("bigP: "+ bigP + " smallP: "+ smallP);
				
				computArrForWord2Vec(ed,100);
			}
		}
		
	}
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		try
		{
			ExperiBasedOnLDAData eb = new ExperiBasedOnLDAData();
			eb.readTopic("C:/Users/jipeng/Desktop/Qiang/dataset/20News/topicWord.txt");
			//eb.readTopic("C:/Users/jipeng/Desktop/Qiang/dataset/20News/topicWord.txt");
			
			//eb.mainForWord2Vec();
			eb.printVector(100);
			//eb.mainForWord2VecGroup();
			
		}catch(Exception e)
		{
			e.printStackTrace();
		}
	}

}
