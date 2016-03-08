
package Data;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import kex.stemmers.MartinPorterStemmer;
import kex.stopwords.Stopwords;
import kex.stopwords.StopwordsEnglish;

public class ProcessData {
	
	//NIPS dataset
	//String dir = "C:/Users/qjp/Desktop/TopicModel/dataset/nipstxt";
	//20 news dataset
	//String dir = "C:/Users/qjp/Desktop/TopicModel/dataset/20_newsgroups";
	String dir = "C:/Users/jipeng/Desktop/Qiang/dataset/BBC/bbc";
	
	int id = 0;
	
	private HashMap<String, Integer> word2IdHash = new HashMap<String, Integer>(); //<word, word's id>
	private HashMap<Integer, String> id2WordHash = new HashMap<Integer, String>(); // <word's id, word>
	private HashMap<Integer, Integer> id2Isf = new HashMap<Integer, Integer>(); // <word's id,  word's isf>
	//ArrayList<String> wordsArr = new ArrayList<String>();

	Pattern p = Pattern.compile("[^a-zA-Z]", Pattern.CASE_INSENSITIVE);
	
	//ArrayList<Integer> wordFreArr = new ArrayList<Integer>();
	
	
	
	ArrayList<ArrayList<Integer>> dataArr = new ArrayList<ArrayList<Integer>>();
	
	//ArrayList<Integer> validTextId = new ArrayList<Integer>();
	
	//private MartinPorterStemmer m_Stemmer = new MartinPorterStemmer();
	
	private Stopwords m_EnStopwords = new StopwordsEnglish();

	MaxentTagger tagger = new MaxentTagger("C:/Users/jipeng/Desktop/TopicModel/stanford-postagger-2014-01-04/models/english-left3words-distsim.tagger");
	
	
	public void getValidText() throws Exception
	{

		File folder = new File(dir);
		File[] listOfFiles = folder.listFiles();

		String path = "C:/Users/jipeng/Desktop/Qiang/dataset/BBC/fileNoStem2/";
		String path2 = "C:/Users/jipeng/Desktop/Qiang/dataset/BBC/fileNoStemPath2/";
		FileWriter fw = new FileWriter("C:/Users/jipeng/Desktop/Qiang/dataset/BBC/Inform.txt");
		
		BufferedWriter bw = new BufferedWriter(fw);
			
		int indexDoc = 0;
		for (File file : listOfFiles) 
		{
		   
			bw.write(file.getName());
			System.out.println(file.getName());
			bw.write(": ");
			bw.newLine();
			File[] nameList = file.listFiles();
			
			FileWriter subfw = new FileWriter(path + file.getName()+".txt");
			
			BufferedWriter subbw = new BufferedWriter(subfw);
			
			FileWriter subfwPath = new FileWriter(path2 + file.getName()+".txt");
			
			BufferedWriter subbwPath = new BufferedWriter(subfwPath);
			
			for(File subFile:nameList)
			{
				bw.write(subFile.getName());
				bw.write("->");
				List<List<HasWord>> sentences = MaxentTagger.tokenizeText(new BufferedReader(new FileReader(subFile.getAbsolutePath())));
				
				//int textLen = 0;
				
				//for (List<HasWord> sentence : sentences)
					//textLen += sentence.size(); 
				
				
				
				
				ArrayList<Integer> textList = new ArrayList<Integer>();
				
				for (List<HasWord> sentence : sentences)
				 {
				      ArrayList<TaggedWord> tSentence = tagger.tagSentence(sentence);
				     
				       
				      for(int j=0; j<tSentence.size(); j++)
				      {
				    	  	String word = tSentence.get(j).value();
			
				    	  //	String token = m_Stemmer.stemString(word);
				    	  	String token = word.toLowerCase();
				    	  	Matcher m = p.matcher(token); // only save these strings only contains characters
				    	  //	if( !m.find()  && token.length()>3 && token.length()<25 )
				    	  	if( Character.isLetter(token.charAt(0))  && token.length()>2 )
					    	{
				    			  if (!m_EnStopwords.isStopword(token)) 
				    			  {
				    				
									  if (word2IdHash.get(token)==null)
									  {
										  
										  textList.add(id);
										
										  word2IdHash.put(token, id);
										  id2WordHash.put(id, token);
										  id2Isf.put(id, 1);
										   
										   id++;
									    } else
									    {
									    	int wid=(Integer)word2IdHash.get(token);
									    	textList.add(wid);	
									    	id2Isf.put(wid, id2Isf.get(wid)+1);
									     }
									    	
									 }
									 
						    	}
				    	  }
				     // textList.add(setenceList);
				    }
				
				if(textList.size()>300 && textList.size()<10000)
				{
					subbw.write(String.valueOf(indexDoc));
					subbw.newLine();
					subbwPath.write(subFile.getAbsolutePath());
					subbwPath.newLine();
				}
				indexDoc++;
				
				bw.write(String.valueOf(sentences.size()));
				bw.newLine();
				dataArr.add(textList);
			}
			
			subbw.close();
			subfw.close();
			subbwPath.close();
			subfwPath.close();
		}
	
		bw.close();
		fw.close();
	
	}

	
	
	public void printTexts() throws Exception
	{
		//int fileI = (int)(percent*100);
		FileWriter fwS = new FileWriter("C:/Users/jipeng/Desktop/Qiang/dataset/BBC/textNoStem4_2000.txt");
		BufferedWriter bwS = new BufferedWriter(fwS);
	
		FileWriter fwW = new FileWriter("C:/Users/jipeng/Desktop/Qiang/dataset/BBC/wordNoStem4_2000.txt");
		BufferedWriter bwW = new BufferedWriter(fwW);
		
		HashMap<Integer,Integer> idInx = new HashMap<Integer,Integer>();
		
		int inx = 0;	
	 
		// ArrayList<Integer> validList = new ArrayList<Integer>();
		 
		 for(int i=0; i<dataArr.size(); i++)
		 {
			 ArrayList<Integer> textList = dataArr.get(i);
		
			 for(int k=0; k<textList.size(); k++)
			 {
				 int sId = textList.get(k);
				 
				 if(id2Isf.get(sId)>4 && id2Isf.get(sId)<2000)
				 {
					 if(!idInx.containsKey(sId))
					 {
						 idInx.put(sId, inx);
						 inx++;
						 bwW.write(id2WordHash.get(sId));
						 bwW.newLine();
					 }
					// validList.add(sId);
				 }
			 }	 
			 
		 }
		
		 bwW.close();
		
		 for(int i=0; i<dataArr.size(); i++)
		 {
			 ArrayList<Integer> textList = dataArr.get(i);
				
			 for(int k=0; k<textList.size(); k++)
			 {
				 int sId = textList.get(k);
				 if(idInx.containsKey(sId))
					 bwS.write(idInx.get(sId)+ " ");
			 }
			 bwS.newLine();;
		 }
		
		bwS.close();
		fwS.close();
		
	}
	
	
	
	
	public static void main(String []args)
	{
		//String path = "C:/Users/qjp/Workspaces/MyEclipse Professional 2014/TopicModel/src/shortLongText/";
		//TextCluster tc = new TextCluster();
		//tc.readFile();
		try
		{
			//double percent = 0.1; //the precentage of long text in all text;
			ProcessData nips = new ProcessData();
			
			nips.getValidText();
			nips.printTexts();
			//nips.printText(percent);
			//nips.printSentences();
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		
	}

}
