package EmbeddingDiver;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;




import kex.stopwords.Stopwords;
import kex.stopwords.StopwordsEnglish;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;



public class EmbeddingDiver2 {

	HashMap<String,Integer> wordIdMap = new HashMap<String,Integer>();
	
	
	//String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/glove.6B.300d.txt";
	String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/vectors.840B.300d.txt";
	//String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/deps.words";
	private int topNSize = 40;
	//float vect[][] = new float[400000][300];
	
	//ArrayList<String> allWord = new ArrayList<String>();
	HashMap<String, float[]> wordMap = new HashMap<String, float[]>();
	
	MaxentTagger tagger = new MaxentTagger("C:/Users/jipeng/Desktop/TopicModel/stanford-postagger-2014-01-04/models/english-left3words-distsim.tagger");
	
	private Stopwords m_EnStopwords = new StopwordsEnglish();
	
	Pattern p = Pattern.compile("[^a-zA-Z]", Pattern.CASE_INSENSITIVE);
	
	DecimalFormat df = new DecimalFormat("0.0000");
	
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
	
	/* the distance between two words */
	public float wordDist(float w1[], float w2[])
	{
		/*float dis = 0;
		
		for(int i=0; i<w1.length; i++)
		{
			dis += (w1[i]-w2[i])*(w1[i]-w2[i]);
		}
		//return dis;
		return  (float)Math.sqrt(dis);*/
		
		return 1-wordSisCosine(w1,w2);
		/*double sis = wordSisCosine(w1,w2);
		
		if(sis<=0)
			return 100;
		else
			return (float)1/(float)sis;*/
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
	
	public ArrayList<Double> normBagOfWords(ArrayList<String> doc)
	{
		ArrayList<Double> normFre = new ArrayList<Double>();
		
		HashMap<String,Integer> wordF = new HashMap<String,Integer>();
		
		for(int i=0; i<doc.size(); i++)
		{
			if(!wordF.containsKey(doc.get(i)))
				wordF.put(doc.get(i), 1);
			else
				wordF.put(doc.get(i), wordF.get(doc.get(i))+1);
		}
		
		for(int i=0; i<doc.size(); i++)
		{
			normFre.add((double)wordF.get(doc.get(i))/doc.size());
		}
		
		return normFre;
	}
	
	public ArrayList<Integer> normBagOfWords(ArrayList<String> doc, ArrayList<String> diff)
	{
		ArrayList<Integer> freArr = new ArrayList<Integer>();
		
		HashMap<String,Integer> wordF = new HashMap<String,Integer>();
		
		for(int i=0; i<doc.size(); i++)
		{
			if(!wordMap.containsKey(doc.get(i)))
				continue;
			if(!wordF.containsKey(doc.get(i)))
			{
				wordF.put(doc.get(i), 1);
				diff.add(doc.get(i));
			}
			else
				wordF.put(doc.get(i), wordF.get(doc.get(i))+1);
		}
		
		for(int i=0; i<diff.size(); i++)
		{
			freArr.add(wordF.get(diff.get(i)));
		}
		
		return freArr;
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
				    	  	if( !m.find()  && token.length()>3 && token.length()<25 )
				    	  //	if( Character.isLetter(token.charAt(0))  && token.length()>3 )
					    	{
				    			  if (!m_EnStopwords.isStopword(token)) 
				    			  {
				    				  partStr.add(token);
				    			  }
					    	}
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
	
	
	
	public double compuDiver2(ArrayList<String> doc)
	{
		double divS = 0.0;
		
		ArrayList<String> diff = new ArrayList<String>();
		
		ArrayList<Integer> freArr = normBagOfWords(doc, diff);
		
		int docLen = doc.size();
		
		ArrayList<Double> dist = new ArrayList<Double>();
		ArrayList<Integer> indArr = new ArrayList<Integer>();
		for(int i=0; i<diff.size(); i++)
			for(int j=i+1; j<diff.size(); j++)
			{
				dist.add((double)wordDist(wordMap.get(diff.get(i)),wordMap.get(diff.get(j))));
				indArr.add(i*10000+j);
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
				if(dist.get(j)>minV)
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
		
		while(docLen>1)
		{
			if(dist.size()<1)
			{	
				
				break;
			}
			double wei = dist.get(0);
			int ind = indArr.get(0);
			int rInd = ind/10000;
			int cInd = ind%10000;
			
			if(freArr.get(rInd)<1 || freArr.get(cInd)<1)
			{
				dist.remove(0);
				indArr.remove(0);
			}else
			{
				int minFre = freArr.get(rInd);
				if(minFre>freArr.get(cInd))
					minFre = freArr.get(cInd);
				
				divS += minFre*wei;
				
				freArr.set(rInd, freArr.get(rInd)-minFre);
				freArr.set(cInd, freArr.get(cInd)-minFre);
				docLen -= minFre*2;
				dist.remove(0);
				indArr.remove(0);
			}
		}
		
		
		//System.out.println("the score of diversity: " + divS);
		
		return divS;//*Math.log(doc.size());
	}
	
	/*public double compuDiver2(String path1,String path2)
	{
		
		ArrayList<String> doc1 = readFile(path1);
		ArrayList<String> doc2 = readFile(path2);
		ArrayList<String> doc = new ArrayList<String>();
		
		doc.addAll(doc1);
		doc.addAll(doc2);
		
		double divS = 0.0;
		
		ArrayList<String> diff = new ArrayList<String>();
		
		ArrayList<Double> normFre = normBagOfWords(doc, diff);
		
		for(int i=0; i<diff.size(); i++)
		{
			//int index1 = allWord.indexOf(diff.get(i));
			
			if(!wordMap.containsKey(diff.get(i)))
				continue;
			
			double dist = 0;
			
			int num  = 0;
			for(int j=0; j<diff.size(); j++)
			{
				if(j==i)
					continue;
				//int index2 = allWord.indexOf(diff.get(j));
				
				if(!wordMap.containsKey(diff.get(j)))
					continue;
				
				//System.out.print(diff.get(i) + " " + diff.get(j) + " ");
				
				double d = wordDist(wordMap.get(diff.get(i)),wordMap.get(diff.get(j)));
				
				//System.out.println(d);
				dist += d;
				num++;
			}
			if(num>0)
				dist /= num;
			
			divS += dist*normFre.get(i);
		}
		
		//divS += divS;
		
		//divS /= diff.size()*(diff.size()-1); 
		
		return divS;
	}
	*/

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
	
	
	
	public double docDist(ArrayList<Double> d1, ArrayList<Double> d2, ArrayList<Double> dist)
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
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		try
		{
			EmbeddingDiver2 ed = new EmbeddingDiver2();
			
			//ed.readVector();
			
			//ed.test();
			//ed.readFile("C:/Users/jipeng/Desktop/dataset/BBC/bbc/business/001.txt");
			//ed.wordDist("", w2)
			//ed.testWordDistance("C:/Users/jipeng/Desktop/dataset/BBC/005.txt");
			//ed.testVector();
			//ed.compuDiverBasedClustering("C:/Users/jipeng/Desktop/dataset/001.txt", "C:/Users/jipeng/Desktop/dataset/002.txt");
			//ed.compuDiverBasedClustering("C:/Users/jipeng/Desktop/dataset/BBC/bbc/business/002.txt", "C:/Users/jipeng/Desktop/dataset/BBC/bbc/business/003.txt");
			//ed.compuDiverBasedClustering("C:/Users/jipeng/Desktop/dataset/BBC/bbc/business/002.txt", "C:/Users/jipeng/Desktop/dataset/BBC/bbc/entertainment/001.txt");
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		
	}

}
