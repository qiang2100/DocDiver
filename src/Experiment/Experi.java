package Experiment;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import kex.stopwords.Stopwords;
import kex.stopwords.StopwordsEnglish;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import EmbeddingDiver.EmbeddingDiver;
import EmbeddingDiver.EmbeddingDiver2;
import EmbeddingDiver.WordClusteringForDiver;
import ldaDiver.LDADiver;
import ldaDiver.PseudoDoc;
import ldaDiver.PseudoDocPath;

public class Experi {
	
		ArrayList<String> pathArr = new ArrayList<String>();
		
		ArrayList<String> pathAbsoArr = new ArrayList<String>();
		
		ArrayList<String> wordArr = new ArrayList<String>();
		
		ArrayList<ArrayList<String>> docArr = new ArrayList<ArrayList<String>>();
		
		//String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/vectors.840B.300d.txt";
		MaxentTagger tagger = new MaxentTagger("C:/Users/jipeng/Desktop/TopicModel/stanford-postagger-2014-01-04/models/english-left3words-distsim.tagger");
		private Stopwords m_EnStopwords = new StopwordsEnglish();
		Pattern p = Pattern.compile("[^a-zA-Z]", Pattern.CASE_INSENSITIVE);
		
		WordClusteringForDiver wcd = new WordClusteringForDiver();
    	
		int right;
		int total;
		
		public Experi()
		{
			wcd.readVector();
		}
		
		public void readFile() throws Exception
		{
			String dir = "C:/Users/jipeng/Desktop/Qiang/dataset/BBC/fileNoStem2/";
			
			File folder = new File(dir);
			File[] listOfFiles = folder.listFiles();

			
			for (File file : listOfFiles)
			{
				pathArr.add(file.getAbsolutePath());
			}
				
		}
		
		public void readAbsoFile() throws Exception
		{
			String dir = "C:/Users/jipeng/Desktop/Qiang/dataset/BBC/fileNoStemPath/";
			
			File folder = new File(dir);
			File[] listOfFiles = folder.listFiles();

			
			for (File file : listOfFiles)
			{
				pathAbsoArr.add(file.getAbsolutePath());
			}
				
		}
	
	   public ArrayList<PseudoDoc> chooseDoc(ArrayList<Integer> index1, ArrayList<Integer> index2)
	    {
	    	ArrayList<PseudoDoc> pseArr = new ArrayList<PseudoDoc>();
	    	
	    	int id = 0;
	    	
	    	boolean isStop = false;
	    	
	    	for(int i=0; i<index1.size(); i++)
	    	{
	    		//if(docArr.get(index1.get(i)).size()<=200)
	    			//continue;
	    		for(int j=i+1; j<index1.size(); j++)
	    		{
	    			//if(docArr.get(index1.get(j)).size()<=200)
		    			//continue;
	    			
	    			//if(docArr.get(index2.get(j%index2.size())).size()<=200)
		    			//continue;
	    			//int ran = (int)(Math.random()*index2.size());
	    			PseudoDoc pse = new PseudoDoc(index1.get(i), index1.get(j),index2.get(j%index2.size()));
	    			//PseudoDoc pse = new PseudoDoc(index1.get(i), index1.get(j),index2.get(0));
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
	    		//if(docArr.get(index2.get(i)).size()<=200)
	    			//continue;
	    		for(int j=i+1; j<index2.size(); j++)
	    		{
	    			//if(docArr.get(index2.get(j)).size()<=200)
		    			//continue;
	    			
	    			//if(docArr.get(index1.get(j%index1.size())).size()<=200)
		    			//continue;
	    			//int ran = (int)(Math.random()*index1.size());
	    			PseudoDoc pse = new PseudoDoc(index2.get(i), index2.get(j),index1.get(j%index1.size()));
	    			//PseudoDoc pse = new PseudoDoc(index2.get(i), index2.get(j),index1.get(0));
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
	   
	   public ArrayList<PseudoDocPath> chooseDocPath(ArrayList<String> index1, ArrayList<String> index2)
	    {
	    	ArrayList<PseudoDocPath> pseArr = new ArrayList<PseudoDocPath>();
	    	
	    	int id = 0;
	    	
	    	boolean isStop = false;
	    	
	    	for(int i=0; i<index1.size(); i++)
	    	{
	    		for(int j=i+1; j<index1.size(); j++)
	    		{
	    			int ran = (int)(Math.random()*index2.size());
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
	    			int ran = (int)(Math.random()*index1.size());
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
	    
	    /* compute the accury of two category */
	    public ArrayList<PseudoDoc> computPse(String path1, String path2) throws Exception 
	    {
	    	ArrayList<Integer> index1 = new ArrayList<Integer>();
	    	
	    	
	    	BufferedReader br1 = new BufferedReader(new FileReader(path1));
			String line = "";
			while ((line = br1.readLine()) != null) {
				int ind = Integer.parseInt(line);
				
				index1.add(ind);
			}
			br1.close();
			
			ArrayList<Integer> index2 = new ArrayList<Integer>();
	    	
	    	
	    	BufferedReader br2 = new BufferedReader(new FileReader(path2));
			//String line = "";
			while ((line = br2.readLine()) != null) {
				int ind = Integer.parseInt(line);
				
				index2.add(ind);
			}
			
			br2.close();
			
			ArrayList<PseudoDoc> pseArr = new ArrayList<PseudoDoc>();
			
			pseArr = chooseDoc(index1,index2);
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
	    
	    public void computAcc(LDADiver lda,String path1, String path2) throws Exception
	    {
	    	ArrayList<PseudoDoc> pseArr = computPse(path1, path2);
	    	
	    	
	    	
	    	int numOfRight = 0;
	    	
			for(int i=0; i<pseArr.size(); i++)
			{
				double div1 = lda.computDivBasedCosine(pseArr.get(i).idDoc1, pseArr.get(i).idDoc2);
				double div2 = lda.computDivBasedCosine(pseArr.get(i).idDoc2, pseArr.get(i).idDoc3);
			//	System.out.println("div1" + div1 + " div2" + div2);
				if(div2>div1)
					numOfRight++;
			}
			
			System.out.println("accury: "+ (double)numOfRight/pseArr.size());
	    }
	    
	  
	   
	   public void readText(String path) throws Exception
	   {
		   BufferedReader br1 = new BufferedReader(new FileReader(path));
			String line = "";
			while ((line = br1.readLine()) != null) {

				String word[] = line.split(" ");
				
				ArrayList<String> doc = new ArrayList<String>();
				
				for(int i=0; i<word.length; i++)
				{
					doc.add(wordArr.get(Integer.parseInt(word[i])));
				}
				docArr.add(doc);
			}
			br1.close();
	   }
	   
	   public void readWord(String path) throws Exception
	   {
		   BufferedReader br1 = new BufferedReader(new FileReader(path));
			String line = "";
			while ((line = br1.readLine()) != null) {

				wordArr.add(line);
			}
			br1.close();
	   }
	   
	   public void readTextAndWord() throws Exception
	   {
		   String path = "C:/Users/jipeng/Desktop/Qiang/dataset/BBC/textNoStem.txt";
		   String wordPath = "C:/Users/jipeng/Desktop/Qiang/dataset/BBC/wordNoStem.txt";
		   
		   readWord(wordPath);
		   
		   readText(path);
	   }
	   
	   public ArrayList<String> merge(int index1, int index2)
	   {
		   ArrayList<String> m = new ArrayList<String>();
		   
		   int threshond = 200;
		   
		   m.addAll(docArr.get(index1).subList(0,threshond));
		  
		   
		   m.addAll(docArr.get(index2).subList(0,threshond));
		   
		   return m;
	   }
	   
	   public void computAccEmbedding(WordClusteringForDiver emb,String path1, String path2) throws Exception
	    {
	    	ArrayList<PseudoDoc> pseArr = computPse(path1, path2);
	    	
	    	//System.out.println(path1);
	    	//System.out.println(path2);
	    	
	    	int numOfRight = 0;
	    	
			for(int i=0; i<pseArr.size(); i++)
			{
				double div1 = emb.compuDiverBasedKmeans(merge(pseArr.get(i).idDoc1,pseArr.get(i).idDoc2));
						
				double div2 = emb.compuDiverBasedKmeans(merge(pseArr.get(i).idDoc2,pseArr.get(i).idDoc3));
				
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
			
			System.out.println((double)numOfRight/pseArr.size());
	    }
	   
	   public void computAccEmbeddingPath(WordClusteringForDiver emb,String path1, String path2) throws Exception
	    {
	    	ArrayList<PseudoDocPath> pseArr = computPsePath(path1, path2);
	    	
	    	//System.out.println(path1);
	    	//System.out.println(path2);
	    	
	    	int numOfRight = 0;
	    	
			for(int i=0; i<pseArr.size(); i++)
			{
				//double div1 = emb.compuDiver2(pseArr.get(i).idDoc1,pseArr.get(i).idDoc2);
				double div1 = emb.testPse(pseArr.get(i).idDoc1,pseArr.get(i).idDoc2);
				//double div2 = emb.compuDiver2(pseArr.get(i).idDoc2,pseArr.get(i).idDoc3);
				double div2 = emb.testPse(pseArr.get(i).idDoc1,pseArr.get(i).idDoc3);
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
	   
	   public void mainFunLDA() throws Exception
	   {
		   System.out.println("LDA");
		   String path = "C:/Users/jipeng/Desktop/Qiang/dataset/BBC/textNoStem2.txt";
	    	int k = 30;
				//int topWords =20; 
	    	LDADiver lda = new LDADiver();
	    	  
	    	lda.callLDA(path,k);
	    	
	    	
	    	readFile();
	    	
	    	//readText(path);
		  
		  for(int i=0; i<pathArr.size()-1; i++)
		  { 
			  System.out.println(pathArr.get(i));
			  System.out.println(pathArr.get(i+1));
			  computAcc(lda, pathArr.get(i),pathArr.get(i+1));
		  }
		 // computAcc(lda, pathArr.get(0),pathArr.get(1));
		 // computAcc(lda, pathArr.get(1),pathArr.get(9));
		 // computAcc(lda, pathArr.get(9),pathArr.get(10));
		 // computAcc(lda, pathArr.get(9),pathArr.get(pathArr.size()-1));
	   }
	   
	   public void mainFunEmb() throws Exception
	   {
		   
	    	readFile();
	    	readTextAndWord();
	    	WordClusteringForDiver emb = new WordClusteringForDiver();
	    	
	    	emb.readVector();
	    	
	    	
		  for(int i=0; i<pathArr.size()-1; i++)
		  { 
			  System.out.println(pathArr.get(i));
			  System.out.println(pathArr.get(i+1));
			  computAccEmbedding(emb, pathArr.get(i),pathArr.get(i+1));
		  }
	    	//computAccEmbedding(emb, pathArr.get(0),pathArr.get(1));
	    	//computAccEmbedding(emb, pathArr.get(1),pathArr.get(2));
	    	//computAccEmbedding(emb, pathArr.get(2),pathArr.get(3));
	    	//computAccEmbedding(emb, pathArr.get(3),pathArr.get(pathArr.size()-1));
	   }
	   
	   public void mainFunEmbPath() throws Exception
	   {
		   
	    	readAbsoFile();
	    	//readTextAndWord();
	    	//EmbeddingDiver emb = new EmbeddingDiver();
	    	//System.out.println("06025: extends");
	    	WordClusteringForDiver wcd = new WordClusteringForDiver();
	    	wcd.readVector();
	    	
	    	
		  for(int i=0; i<pathAbsoArr.size()-1; i++)
		  { 
			  System.out.println(pathAbsoArr.get(i));
			  for(int j=i+1; j<pathAbsoArr.size(); j++)
			  {
				  System.out.println(pathAbsoArr.get(j));
				  computAccEmbeddingPath(wcd, pathAbsoArr.get(i),pathAbsoArr.get(j));
			  }
		  }
	    	//computAccEmbedding(emb, pathArr.get(0),pathArr.get(1));
	    	//computAccEmbedding(emb, pathArr.get(1),pathArr.get(2));
	    	//computAccEmbedding(emb, pathArr.get(2),pathArr.get(3));
	    	//computAccEmbedding(emb, pathArr.get(3),pathArr.get(pathArr.size()-1));
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
					    	  	if( !m.find()  && token.length()>=3 && token.length()<25 )
					    	  //	if( Character.isLetter(token.charAt(0))  && token.length()>2 )
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
	   
	  //path1 and path2 from two different clustering
	   public void computDiverForTwoDiffPath(String dir1, String dir2, BufferedWriter bw, WordClusteringForDiver wcd) throws Exception
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
			
			System.out.println("WordEmbedding");
			System.out.println(dir1);
			System.out.println(dir2);
			bw.write(dir1);
			bw.newLine();
			bw.write(dir2);
			bw.newLine();
			
			
	    	
	    	int total = 0;
	    	
	    	int right = 0;
	    	
			for(int i=0; i<path1Arr.size()-1; i++)
			{
				if(path1Arr.get(i).size()<=200 || path1Arr.get(i+1).size()<=200)
					continue;
				
				ArrayList<String> doc1 = new ArrayList<String>();
				
				doc1.addAll(path1Arr.get(i).subList(0, 200));
				doc1.addAll(path1Arr.get(i+1).subList(0, 200));
				double div1 = wcd.compuDiverBasedKmeans(doc1);
				
				for(int j=0; j<path2Arr.size(); j++)
				{
					if(path2Arr.get(j).size()<=200)
						continue;
					ArrayList<String> doc2 = new ArrayList<String>();
					
					doc2.addAll(path1Arr.get(i).subList(0, 200));
					doc2.addAll(path2Arr.get(j).subList(0, 200));
					double div2 = wcd.compuDiverBasedKmeans(doc2);
					
					if(div2>div1)
						right++;
					total++;
				}
			}
			
			double acc = (double)right/total;
			System.out.println("-------------" + (double)right/total);
			bw.write(String.valueOf(acc));
			bw.newLine();
			
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
			
			System.out.println("WordEmbedding");
			System.out.println(dir1);
			System.out.println(dir2);
		
			
			
	    //	int total = 0;
	    	
	    	//int right = 0;
	    	
//			String s2 = "relationship";
//			float []v2 = wcd.wordMap.get(s2);
//			for(int ii=0; ii<v2.length; ii++)
//				System.out.print(v2[ii]+ " ");
//			System.out.println();
			for(int i=0; i<path1Arr.size()-1; i++)
			{
				if(path1Arr.get(i).size()<=200 || path1Arr.get(i+1).size()<=200)
					continue;
				
				ArrayList<String> doc1 = new ArrayList<String>();
				
				doc1.addAll(path1Arr.get(i).subList(0, 200));
				doc1.addAll(path1Arr.get(i+1).subList(0, 200));
				//System.out.println(" i " + i+" ");
				//if(i==5)
			    //System.out.println(doc1.toString());
				
				
				
				
				//double div1 = wcd.compuDiverBasedKmeans(doc1);
				double div1 = wcd.compuDiver(doc1);
				/*float []v3 = wcd.wordMap.get(s2);
				for(int ii=0; ii<v3.length; ii++)
					System.out.print(v3[ii]+ " ");
				System.out.println();*/
				 //System.out.println("i " + div1 + " " + doc1.toString());
				for(int j=0; j<path2Arr.size(); j++)
				{
					//if(j==12)
					   //System.out.println(" j " + j+" ");
					if(path2Arr.get(j).size()<=200)
						continue;
					ArrayList<String> doc2 = new ArrayList<String>();
					
					doc2.addAll(path1Arr.get(i).subList(0, 200));
					doc2.addAll(path2Arr.get(j).subList(0, 200));
					//System.out.println(doc2.toString());
					//double div2 = wcd.compuDiverBasedKmeans(doc2);
					double div2 = wcd.compuDiver(doc2);
					//System.out.println("j " + div2 + " " + doc2.toString());
					if(div2>div1)
						right++;
					/*else{
						if(div1-div2>0.03)
						{
							System.out.println(i + " " +div1 + " " + j + " "+ div2);
							//wcd.compuDiverBasedKmeans(doc1);
							//wcd.compuDiverBasedKmeans(doc2);
						}
					}*/
					total++;
				}
			}
			
			System.out.println("-------------" + right+ " " +total);
			
			/*float []v3 = wcd.wordMap.get(s2);
			for(int ii=0; ii<v3.length; ii++)
				System.out.print(v3[ii]+ " ");
			System.out.println();*/
			for(int i=0; i<path2Arr.size()-1; i++)
			{
				if(path2Arr.get(i).size()<=200 || path2Arr.get(i+1).size()<=200)
					continue;
				
				ArrayList<String> doc1 = new ArrayList<String>();
				
				doc1.addAll(path2Arr.get(i).subList(0, 200));
				doc1.addAll(path2Arr.get(i+1).subList(0, 200));
				//System.out.println(" i " + i+" ");
				//if(i==5)
				//System.out.println(doc1.toString());
				
				double div1 = wcd.compuDiver(doc1);
				//double div1 = wcd.compuDiverBasedKmeans(doc1);
				
				//String s2 = "relationship";
				
				//System.out.println("i " + div1 + " " + doc1.toString());
				for(int j=0; j<path1Arr.size(); j++)
				{
					//System.out.println(" j " + j+" ");
					if(path1Arr.get(j).size()<=200)
						continue;
					ArrayList<String> doc2 = new ArrayList<String>();
					
					doc2.addAll(path2Arr.get(i).subList(0, 200));
					doc2.addAll(path1Arr.get(j).subList(0, 200));
					//System.out.println(doc2.toString());
					//double div2 = wcd.compuDiverBasedKmeans(doc2);
					double div2 = wcd.compuDiver(doc2);
					//System.out.println("j " + div2 + " " + doc2.toString());
					if(div2>div1)
						right++;
					/*else{
						if(div1-div2>0.03)
						{
							System.out.println(i + " " +div1 + " " + j + " "+ div2);
							//wcd.compuDiverBasedKmeans(doc1);
							//wcd.compuDiverBasedKmeans(doc2);
						}
					}*/
					total++;
				}
			}
			
			double acc = (double)right/total;
			System.out.println("-------------" + right+ " " +total);
			System.out.println("-------------" + (double)right/total);
			
			
	   }
	   
	   public void mianFunDirPath() throws Exception
	   {
		   
		   String dir1 = "C:/Users/jipeng/Desktop/dataset/GoogleNew10052015/0/91";
		   String dir2 = "C:/Users/jipeng/Desktop/dataset/GoogleNew10052015/2/131";
		   	
		   computDiverForTwoDiffPath(dir2,dir1);
	   }
	   
	   public void AllCataPath() throws Exception
	   {
		   String dir = "C:/Users/jipeng/Desktop/dataset/GoogleNew10062015/";
		  
		   
		   File folder = new File(dir);
			File[] listOfFiles = folder.listFiles();
			
			ArrayList<String> pathArr = new ArrayList<String>();
			for (File file : listOfFiles) 
			{
				pathArr.add(file.getAbsolutePath());
			}
			
			FileWriter fw = new FileWriter("Result/Ebedding_20Topics_GoogleNewsAcc.txt");
			BufferedWriter bw = new BufferedWriter(fw);
			
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
	   
	   public void mianFunDirPathGroup() throws Exception
	   {
		   
		   
		   String dir = "C:/Users/jipeng/Desktop/dataset/GoogleNew10052015/All";
		 //  String dir2 = "C:/Users/jipeng/Desktop/dataset/GoogleNew10052015/";
		   //computDiverForTwoDiffPath(dir1,dir2);
		   
		   FileWriter fw = new FileWriter("C:/Users/jipeng/Desktop/Qiang/EMB0.75.txt");
			
			BufferedWriter bw = new BufferedWriter(fw);
			
		   File folder = new File(dir);
			File[] listOfFiles = folder.listFiles();
			
			WordClusteringForDiver wcd = new WordClusteringForDiver();
	    	wcd.readVector();
	    	
			
			for (int i=0; i<listOfFiles.length-1; i++) 
			{
				computDiverForTwoDiffPath(listOfFiles[i].getAbsolutePath(),listOfFiles[i+1].getAbsolutePath(),bw, wcd);
			}
			
			bw.close();
	   }
	   
	    public static void main(String []args)
	    {
	    	try
	    	{
	    		Experi e = new Experi();
	    		
	    		//e.mainFunLDA();
	    		e.mainFunEmb();
	    		//e.mainFunEmbPath();
	    		//e.mianFunDirPath();
	    		//e.mianFunDirPathGroup();
	    		//e.AllCataPath();
	    	}catch(Exception e)
	    	{
	    		e.printStackTrace();
	    	}
	    	

	    }

}
