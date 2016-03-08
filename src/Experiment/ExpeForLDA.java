package Experiment;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import kex.stopwords.Stopwords;
import kex.stopwords.StopwordsEnglish;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import EmbeddingDiver.WordClusteringForDiver;
import ldaDiver.LDADiver;
import ldaDiver.PseudoDocPath;

public class ExpeForLDA {

	ArrayList<String> wordArr = new ArrayList<String>();
	
	ArrayList<ArrayList<String>> topic2Arr = new ArrayList<ArrayList<String>>();
	
	ArrayList<String> pathAbsoArr = new ArrayList<String>();
	double [][]phi;
	double [][]dist;
	
	int []z;
	int nt[];
	double alpha = 0.1;
	
	int right;
	int total;
	MaxentTagger tagger = new MaxentTagger("C:/Users/jipeng/Desktop/TopicModel/stanford-postagger-2014-01-04/models/english-left3words-distsim.tagger");
	
	private Stopwords m_EnStopwords = new StopwordsEnglish();
	
	Pattern p = Pattern.compile("[^a-zA-Z]", Pattern.CASE_INSENSITIVE);
	
	 public void readWord(String path) throws Exception
	 {
		   BufferedReader br1 = new BufferedReader(new FileReader(path));
			String line = "";
			while ((line = br1.readLine()) != null) {

			wordArr.add(line);
			}
			br1.close();
	  }
	
	 public void savePhi(double [][]phi) throws Exception
	 {
		 String phiPath = "C:/Users/jipeng/Desktop/dataset/nytimes/phi100.txt";
		 
		 FileWriter fw = new FileWriter(phiPath);
			BufferedWriter bw = new BufferedWriter(fw);
			
		 for(int i=0; i<phi.length; i++)
		 {
			 String oneT = "";
			 for(int j=0; j<phi[i].length; j++)
			 {
				 oneT += Double.toString(phi[i][j])+ " ";
			 }
			 bw.write(oneT);
			 bw.newLine();
		 }
		 bw.close();
		 fw.close();
	 }
	 
	 public void loadPhi(String path) throws Exception
	 {
		 	BufferedReader br1 = new BufferedReader(new FileReader(path));
			String line = "";
			int numT = 0;
			while ((line = br1.readLine()) != null) {

				String []tPhi = line.split(" ");
				
				for(int i=0; i<tPhi.length; i++)
					phi[numT][i] = Double.parseDouble(tPhi[i]);
				
				numT++;
			}
			
			br1.close();
			
	 }
	 
	 public void saveDistForTopics(LDADiver lda, int k) throws Exception
	 {
		 String phiPath = "C:/Users/jipeng/Desktop/dataset/nytimes/topicDist100.txt";
		 
		 FileWriter fw = new FileWriter(phiPath);
			BufferedWriter bw = new BufferedWriter(fw);
			
		 for(int i=0; i<k; i++)
		 {
			 String oneT = "";
			 for(int j=0; j<k; j++)
			 {
				 oneT += Double.toString(lda.getDistanceBasedCosine(i, j))+ " ";
			 }
			 bw.write(oneT);
			 bw.newLine();
		 }
		 bw.close();
		 fw.close();
	 }
	 
	 public void loadSisOfTopics(String path) throws Exception
	 {
		 	BufferedReader br1 = new BufferedReader(new FileReader(path));
			String line = "";
			int numT = 0;
			while ((line = br1.readLine()) != null) {

				String []sisT = line.split(" ");
				
				for(int i=0; i<sisT.length; i++)
					dist[numT][i] = Double.parseDouble(sisT[i]);
				numT++;
			}
			br1.close();
			
	 }
	 
	 
	public void mainFunLDA() throws Exception
	   {
		  
		   String path = "C:/Users/jipeng/Desktop/dataset/nytimes/nytimes2.txt";
		   //String wordPath = "C:/Users/jipeng/Desktop/dataset/nytimes/nytimes2Word.txt";
	    	int k = 100;
	    	 System.out.println("LDA: the number of topic: " + k);
				//int topWords =20; 
	    	LDADiver lda = new LDADiver();
	    	  
	    	lda.callLDA(path,k);
	    	
	    	savePhi(lda.getPhi());
	    	saveDistForTopics(lda,k);
	    	//readWord(wordPath);
	   }
	
	private int sampleFullConditional(ArrayList<Integer> textId, int n,int K) {

        // remove z_i from the count variables
        int topic = z[n];
        //nw[sData.get(m).get(n)][topic]--;
        nt[topic]--;
       // nwsum[topic]--;
        //ndsum[m]--;

        // do multinomial sampling via cumulative method:
        double[] p = new double[K];
        for (int k = 0; k < K; k++) {
            p[k] = phi[k][textId.get(n)]* (nt[k] + alpha);// / (ndsum[m] + K * alpha);
        }
        // cumulate multinomial parameters
        for (int k = 1; k < p.length; k++) {
            p[k] += p[k - 1];
        }
        // scaled sample because of unnormalised p[]
        double u = Math.random() * p[K - 1];
        for (topic = 0; topic < p.length; topic++) {
            if (u < p[topic])
                break;
        }

        // add newly estimated z_i to count variables
       // nw[sData.get(m).get(n)][topic]++;
        nt[topic]++;
        //nwsum[topic]++;
       // ndsum[m]++;

        return topic;
    }

	
	public int[] trainTheta(ArrayList<String> text, int k)
	{
		ArrayList<Integer> textId = new ArrayList<Integer>();
		
		for(int i=0; i<text.size(); i++)
		{
			int id = wordArr.indexOf(text.get(i));
			
			if(id!=-1)
				textId.add(id);
		}
		
		z = new int[textId.size()];
		
		nt = new int[k];
		
		for(int i=0; i<z.length; i++)
		{
			z[i] = (int)(Math.random()*k);
			nt[z[i]]++;
		}
		
		
		int maxIt =100;
		int it = 0;
		while(it<maxIt)
		{
			for(int i=0; i<z.length; i++)
			{
				int topic = sampleFullConditional(textId,i,k);
				z[i] = topic;
			}
			it++;
		}
		
		return nt;
	}
	
	public double computScore(int K,int totalWord)
	{
		double div = 0.0;
		for(int i=0; i<K; i++)
    	{
    		for(int j=0; j<K; j++)
    		{
    			div += ((double)nt[i]/totalWord)*((double)nt[j]/totalWord)*(dist[i][j]);
    			/*if(i==j)
    				System.out.println("i==j" + getDistanceBasedCosine(i,j));
    			else
    				System.out.println("i!=j" + getDistanceBasedCosine(i,j));*/
    		}
    	}
		return div;
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
	
	public double compuDiverScore(ArrayList<String> doc, int k)
	{
		trainTheta(doc,k);
		return computScore(k,doc.size());	
	}
	
	
	
	public void readAbsoFile(String dir) throws Exception
	{
		
		
		File folder = new File(dir);
		File[] listOfFiles = folder.listFiles();

		
		for (File file : listOfFiles)
		{
			pathAbsoArr.add(file.getAbsolutePath());
		}
			
	}
	
	public ArrayList<PseudoDocPath> chooseDocPath(ArrayList<String> index1, ArrayList<String> index2)
    {
    	ArrayList<PseudoDocPath> pseArr = new ArrayList<PseudoDocPath>();
    	
    	int id = 0;
    	
    	boolean isStop = false;
    	
    	for(int i=0; i<index1.size(); i++)
    	{
    		for(int j=i+1; j<index1.size(); j++)
    		{
    			//int ran = (int)(Math.random()*index2.size());
    			PseudoDocPath pse = new PseudoDocPath(index1.get(i), index1.get(j),index2.get(j%index2.size()));
    			
    			pseArr.add(pse);
    			if((++id)==50)
    			{
    				isStop = true;
    				break;
    			}
    		}
    		if(isStop)
    			break;
    	}
    	
    	isStop = false;
    	
    	for(int i=0; i<index2.size(); i++)
    	{
    		for(int j=i+1; j<index2.size(); j++)
    		{
    			//int ran = (int)(Math.random()*index1.size());
    			PseudoDocPath pse = new PseudoDocPath(index2.get(i), index2.get(j),index1.get(j%index1.size()));
    			pseArr.add(pse);
    			if((++id)==100)
    			{
    				isStop = true;
    				break;
    			}
    		}
    		if(isStop)
    			break;
    	}
    	
    	
    	return pseArr;
    }
	
	public ArrayList<PseudoDocPath> computPsePath(String path1, String path2) throws Exception 
    {
    	ArrayList<String> index1 = new ArrayList<String>();
    	
    	
    	BufferedReader br1 = new BufferedReader(new FileReader(path1));
		String line = "";
		while ((line = br1.readLine()) != null) {
			//int ind = Integer.parseInt(line);
			
			index1.add(line);
		}
		br1.close();
		
		ArrayList<String> index2 = new ArrayList<String>();
    	
    	
    	BufferedReader br2 = new BufferedReader(new FileReader(path2));
		//String line = "";
		while ((line = br2.readLine()) != null) {
			//int ind = Integer.parseInt(line);
			
			index2.add(line);
		}
		
		br2.close();
		
		ArrayList<PseudoDocPath> pseArr = new ArrayList<PseudoDocPath>();
		
		pseArr = chooseDocPath(index1,index2);
		return pseArr;
		
    }
	
	public ArrayList<String> readFile(String path)
	{
		try{
			
			ArrayList<String> partStr = new ArrayList<String>();
			
			List<List<HasWord>> sentences = MaxentTagger.tokenizeText(new BufferedReader(new FileReader(path)));
			
			//int textLen = 0;
			
			//for (List<HasWord> sentence : sentences)
				//textLen += sentence.size(); 
			ArrayList<Integer> textList = new ArrayList<Integer>();
			
			for (List<HasWord> sentence : sentences)
			{
				 ArrayList<TaggedWord> tSentence = tagger.tagSentence(sentence);
			     
				 //System.out.println(tSentence.toString());
			       
			      for(int j=0; j<tSentence.size(); j++)
			      {
			    	 // String tag = tSentence.get(j).tag();
			    	  
			    	  // System.out.println(tag+ " "+ tSentence.get(j).value());
			    	  //if(tag.length()>=2 && tag.substring(0, 2).equals("NN"))
			    	  //{
			    		  String word = tSentence.get(j).value();
			  			
				    	  //	String token = m_Stemmer.stemString(word);
				    	  	String token = word.toLowerCase();
			    		 // String token = word;
				    	  	Matcher m = p.matcher(token); // only save these strings only contains characters
				    	  	if( !m.find()  && token.length()>2 && token.length()<25 )
				    	  	//if( Character.isLetter(token.charAt(0))  && token.length()>2 )
					    	//{
				    			  if (!m_EnStopwords.isStopword(token)) 
				    			  {
				    				  partStr.add(token);
				    			  }
					    	//}
			    	  	//System.out.println();
			    	  	//String word = tSentence.get(j).value();
			    	  //}
			      }
			}
			return partStr;
			
			
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		System.out.println("Warning!");
		return null;
	}
	
	public ArrayList<String> combinePse(String path1, String path2)throws Exception
	{
		ArrayList<String> doc1 = readFile(path1);
		ArrayList<String> doc2 = readFile(path2);
		ArrayList<String> doc = new ArrayList<String>();
		
		doc.addAll(doc1.subList(0, 50));
		doc.addAll(doc2.subList(0, 50));
		
	
		//System.out.println(div);
		return doc;
	}
	
	 public void computAccPath(String path1, String path2, int k) throws Exception
	    {
	    	ArrayList<PseudoDocPath> pseArr = computPsePath(path1, path2);
	    	
	    	//System.out.println(path1);
	    	//System.out.println(path2);
	    	
	    	int numOfRight = 0;
	    	
			for(int i=0; i<pseArr.size(); i++)
			{
				//double div1 = emb.compuDiver2(pseArr.get(i).idDoc1,pseArr.get(i).idDoc2);
				double div1 = compuDiverScore(combinePse(pseArr.get(i).idDoc1,pseArr.get(i).idDoc2),k);
				//double div2 = emb.compuDiver2(pseArr.get(i).idDoc2,pseArr.get(i).idDoc3);
				double div2 = compuDiverScore(combinePse(pseArr.get(i).idDoc2,pseArr.get(i).idDoc3),k);
				if(div2>div1)
				{
					numOfRight++;
					//System.out.println("score: "+ div1 + " " + div2 );
				}
				else
				{
					//int len = docArr.get(pseArr.get(i).idDoc1).size() + docArr.get(pseArr.get(i).idDoc2).size();
					
					//int len2 = docArr.get(pseArr.get(i).idDoc2).size() + docArr.get(pseArr.get(i).idDoc3).size();
					//System.out.println("score: "+ div1 + " " + div2 );
				}
			}
			
			System.out.println("-------------" + (double)numOfRight/pseArr.size());
	    }
	
	 public void computDiv() throws Exception
	   {
		   
		   String wordPath = "C:/Users/jipeng/Desktop/dataset/nytimes/nytimes2Word.txt";
		   readWord(wordPath);
		   
		   int k =10;
		   int V = wordArr.size();
		   phi = new double[k][V];
		   
		   String phiPath = "C:/Users/jipeng/Desktop/dataset/nytimes/phi10.txt";
		   loadPhi(phiPath);
		   String distPath = "C:/Users/jipeng/Desktop/dataset/nytimes/topicDist10.txt";
		   dist = new double[k][k];
		   loadSisOfTopics(distPath);
		   
		   int num = 100;
		   
		   readTopic("C:/Users/jipeng/Desktop/Qiang/dataset/20News/topicWord.txt");
		   System.out.println("LDA");
		   int right = 0;
		   int total = 0;
		   
		   for(int i=0; i<topic2Arr.size(); i++)
			{
				ArrayList<String> oneTopic = new ArrayList<String>();
				oneTopic.addAll(topic2Arr.get(i).subList(0, 2*num));
				
				
				double oneScore = compuDiverScore(oneTopic,k);
				
				//System.out.println("------------------------------");
				//System.out.println("the " + i + " oneScore: " + oneScore);
				
				for(int j=0; j<topic2Arr.size(); j++)
				{
					if(i==j)
						continue;
					ArrayList<String> twoTopic = new ArrayList<String>();
					twoTopic.addAll(topic2Arr.get(i).subList(0, num));
					twoTopic.addAll(topic2Arr.get(j).subList(0, num));
					
					double twoScore = compuDiverScore(twoTopic,k);
					
					//System.out.println("the " + j + " twoScore: " + twoScore);
					
					total++;
					if(twoScore>oneScore)
						right++;
				}
				
			}
				//int topWords =20; 
	    	
	    	//readWord(wordPath);
		   System.out.println("Accury: " + (double)right/total);
	   }
	 
	 public void mainFunLDAPath() throws Exception
	   {
		   
		 	String dir = "C:/Users/jipeng/Desktop/Qiang/dataset/BBC/fileNoStemPath/";
	    	readAbsoFile(dir);
	    	
	    	 String wordPath = "C:/Users/jipeng/Desktop/dataset/nytimes/nytimes2Word.txt";
			   readWord(wordPath);
			   
			   int k = 30;
			   int V = wordArr.size();
			   phi = new double[k][V];
			   
			   String phiPath = "C:/Users/jipeng/Desktop/dataset/nytimes/phi30.txt";
			   loadPhi(phiPath);
			   String sisPath = "C:/Users/jipeng/Desktop/dataset/nytimes/topicDist30.txt";
			   dist = new double[k][k];
			   loadSisOfTopics(sisPath);
			   
			 //  int num = 50;

	    	
		  for(int i=0; i<pathAbsoArr.size()-1; i++)
		  { 
			  System.out.println(pathAbsoArr.get(i));
			  System.out.println(pathAbsoArr.get(i+1));
			  computAccPath(pathAbsoArr.get(i),pathAbsoArr.get(i+1),k);
		  }
	    	//computAccEmbedding(emb, pathArr.get(0),pathArr.get(1));
	    	//computAccEmbedding(emb, pathArr.get(1),pathArr.get(2));
	    	//computAccEmbedding(emb, pathArr.get(2),pathArr.get(3));
	    	//computAccEmbedding(emb, pathArr.get(3),pathArr.get(pathArr.size()-1));
	   }
	 
	//path1 and path2 from two different clustering
	   public void computDiverForTwoDiffPath(String dir1, String dir2) throws Exception
	   {
		   File folder = new File(dir1);
			File[] listOfFiles = folder.listFiles();
			
			ArrayList<ArrayList<String>> path1Arr = new ArrayList<ArrayList<String>>();
			for (File file : listOfFiles) 
			{
				path1Arr.add(readFile(file.getAbsolutePath()));
			}
			
			File folder2 = new File(dir2);
			File[] listOfFiles2 = folder2.listFiles();
			
			ArrayList<ArrayList<String>> path2Arr = new ArrayList<ArrayList<String>>();
			for (File file : listOfFiles2) 
			{
				path2Arr.add(readFile(file.getAbsolutePath()));
			}
			System.out.println("LDA _ 30");
			System.out.println(dir1);
			System.out.println(dir2);
			
			int k =30;
			for(int i=0; i<path1Arr.size()-1; i++)
			{
				if(path1Arr.get(i).size()<=200 || path1Arr.get(i+1).size()<=200)
					continue;
				
				ArrayList<String> doc1 = new ArrayList<String>();
				
				doc1.addAll(path1Arr.get(i).subList(0, 200));
				doc1.addAll(path1Arr.get(i+1).subList(0, 200));
				double div1 = compuDiverScore(doc1,k);
				
				for(int j=0; j<path2Arr.size(); j++)
				{
					if(path2Arr.get(j).size()<=200)
						continue;
					
					ArrayList<String> doc2 = new ArrayList<String>();
					
					doc2.addAll(path1Arr.get(i).subList(0, 200));
					doc2.addAll(path2Arr.get(j).subList(0, 200));
					double div2 = compuDiverScore(doc2,k);
					
					if(div2>div1)
						right++;
					total++;
				}
			}
			System.out.println("-------------right " + right + " total "+ total);
			//double acc = (double)right/total;
			
			for(int i=0; i<path2Arr.size()-1; i++)
			{
				if(path2Arr.get(i).size()<=200 || path2Arr.get(i+1).size()<=200)
					continue;
				
				ArrayList<String> doc1 = new ArrayList<String>();
				
				doc1.addAll(path2Arr.get(i).subList(0, 200));
				doc1.addAll(path2Arr.get(i+1).subList(0, 200));
				double div1 = compuDiverScore(doc1,k);
				
				for(int j=0; j<path1Arr.size(); j++)
				{
					if(path1Arr.get(j).size()<=200)
						continue;
					
					ArrayList<String> doc2 = new ArrayList<String>();
					
					doc2.addAll(path2Arr.get(i).subList(0, 200));
					doc2.addAll(path1Arr.get(j).subList(0, 200));
					double div2 = compuDiverScore(doc2,k);
					
					if(div2>div1)
						right++;
					total++;
				}
			}
			System.out.println("-------------right " + right + " total "+ total);
			System.out.println("-------------" + (double)right/total);
			
	   }
	   
	   
	   
	   public void AllCataPath() throws Exception
	   {
		   
		   String wordPath = "C:/Users/jipeng/Desktop/dataset/nytimes/nytimes2Word.txt";
		   readWord(wordPath);
		   
		   int k = 30;
		   int V = wordArr.size();
		   phi = new double[k][V];
		   
		   String phiPath = "C:/Users/jipeng/Desktop/dataset/nytimes/phi30.txt";
		   loadPhi(phiPath);
		   String sisPath = "C:/Users/jipeng/Desktop/dataset/nytimes/topicDist30.txt";
		   dist = new double[k][k];
		   loadSisOfTopics(sisPath);
		   
		   
		   String dir = "C:/Users/jipeng/Desktop/dataset/GoogleNew10062015/";
		  
		   
		   File folder = new File(dir);
			File[] listOfFiles = folder.listFiles();
			
			ArrayList<String> pathArr = new ArrayList<String>();
			for (File file : listOfFiles) 
			{
				pathArr.add(file.getAbsolutePath());
			}
			
			FileWriter fw = new FileWriter("Result/LDA_Topic30_GoogleNewsAcc.txt");
			BufferedWriter bw = new BufferedWriter(fw);
			
			System.out.println("LDA");
			
			for(int i=0; i<pathArr.size(); i++)
			{
				for(int j=0; (j!=i) && j<pathArr.size(); j++)
				{
					bw.write(pathArr.get(i));
					bw.newLine();
					bw.write(pathArr.get(j));
					bw.newLine();
					double acc = twoCataPath(pathArr.get(i),pathArr.get(j));
					bw.write(String.valueOf(acc));
					bw.newLine();
				}
			}
			
			bw.close();
			fw.close();
			
	   }
	   
	   public double twoCataPath(String dir1, String dir2) throws Exception
	   {
		   File folder = new File(dir1);
			File[] listOfFiles = folder.listFiles();
			
			ArrayList<String> path1Arr = new ArrayList<String>();
			for (File file : listOfFiles) 
			{
				path1Arr.add(file.getAbsolutePath());
			}
			
			File folder2 = new File(dir2);
			File[] listOfFiles2 = folder2.listFiles();
			
			ArrayList<String> path2Arr = new ArrayList<String>();
			for (File file : listOfFiles2) 
			{
				path2Arr.add(file.getAbsolutePath());
			}
			
			 right= 0;
			 total = 0;
			
			for(int i=0; i<path1Arr.size(); i++)
			{
				for(int j=0; j<path2Arr.size(); j++)
				{
					
					computDiverForTwoDiffPath(path1Arr.get(i),path2Arr.get(j));
				}
			}
			
			if(total==0)
				return 0;
			return (double)right/total;
	   }
	   
	   
	   public void mianFunDirPath() throws Exception
	   {
		   String wordPath = "C:/Users/jipeng/Desktop/dataset/nytimes/nytimes2Word.txt";
		   readWord(wordPath);
		   
		   int k = 30;
		   int V = wordArr.size();
		   phi = new double[k][V];
		   
		   String phiPath = "C:/Users/jipeng/Desktop/dataset/nytimes/phi30.txt";
		   loadPhi(phiPath);
		   String sisPath = "C:/Users/jipeng/Desktop/dataset/nytimes/topicDist30.txt";
		   dist = new double[k][k];
		   loadSisOfTopics(sisPath);
		   
		  // String dir1 = "C:/Users/jipeng/Desktop/dataset/GoogleNew10062015/Entert/109";
		 //  String dir2 = "C:/Users/jipeng/Desktop/dataset/GoogleNew10062015/Business/96";
		   String dir1 = "C:/Users/jipeng/Desktop/dataset/GoogleNew10052015/0/91";
		   String dir2 = "C:/Users/jipeng/Desktop/dataset/GoogleNew10052015/2/131";
		   computDiverForTwoDiffPath(dir1,dir2);
	   }
	   
	   
	   public void computForOneText() throws Exception
	   {
		   String path = "C:/Users/jipeng/Desktop/Qiang/1.txt";
		   ArrayList<String> text = readFile(path);
		   
		   String path2 = "C:/Users/jipeng/Desktop/Qiang/2.txt";
		   ArrayList<String> text2 = readFile(path2);
		   
		   String wordPath = "C:/Users/jipeng/Desktop/dataset/nytimes/nytimes2Word.txt";
		   readWord(wordPath);
		   
		   int k = 30;
		   int V = wordArr.size();
		   phi = new double[k][V];
		   
		   String phiPath = "C:/Users/jipeng/Desktop/dataset/nytimes/phi30.txt";
		   loadPhi(phiPath);
		   String sisPath = "C:/Users/jipeng/Desktop/dataset/nytimes/topicDist30.txt";
		   dist = new double[k][k];
		   loadSisOfTopics(sisPath);
		   
		   double div1 = compuDiverScore(text,k);
		   
		   double div2 = compuDiverScore(text2,k);
		   
		   System.out.println(div1 + " " + div2);
	   }
	
	 public static void main(String[] args) {

 		
       //ArrayList<Integer> classes = new ArrayList<Integer>();
      // int a[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};
      // classes.add(2);
       //String path="C:/Users/qjp/Workspaces/MyEclipse Professional 2014/TopicModel/src/shortLongText/title2P0.txt";
 		//String path="C:/Users/qjp/Workspaces/MyEclipse Professional 2014/TopicModel/src/shortLongText/newsGooglecontent.txt";
 		
       //lda.mainFun(path, 117);
 		try
 		{
 			  //String path = "C:/Users/qjp/Desktop/TopicModel/dataset/title.txt";
 			 
 			  //String wordPath = "C:/Users/qjp/Desktop/TopicModel/dataset/NIPS/word.txt";
 			  //String lablePath = "C:/Users/qjp/Desktop/TopicModel/dataset/NIPS/lable.txt";
 			  
 			//  lda.mainFun();
 			  
 			 // lda.percentIterMain();
 	          //lda.mainFun(path, lablePath, k);
 			// lda.iterMain(path, wordPath);
 			
 			ExpeForLDA lda = new ExpeForLDA();
 			lda.computDiv(); //compute the diversity based on the topic extracted by LDA.
 			//lda.mainFunLDA(); // call LDA, train LDA model and save phi and sis.
 			//lda.mainFunLDAPath(); // comput the diversity from two path of two documents.
 			//lda.mianFunDirPath();
 			//lda.AllCataPath();
 			//lda.computForOneText();
 		}catch(Exception e)
 		{
 			e.printStackTrace();
 		}
     
 	
 }

}
