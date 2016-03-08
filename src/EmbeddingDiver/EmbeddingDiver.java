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

import textcluster.TermVector;

import com.telmomenezes.jfastemd.JFastEMD;

import kex.stopwords.Stopwords;
import kex.stopwords.StopwordsEnglish;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;



public class EmbeddingDiver {

	HashMap<String,Integer> wordIdMap = new HashMap<String,Integer>();
	
	
	String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/glove.6B.300d.txt";
	//String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/vectors.840B.300d.txt";
	//String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/deps.words";
	private int topNSize = 200;
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
	
	public ArrayList<Double> normBagOfWords(ArrayList<String> doc, ArrayList<String> diff)
	{
		ArrayList<Double> normFre = new ArrayList<Double>();
		
		HashMap<String,Integer> wordF = new HashMap<String,Integer>();
		
		for(int i=0; i<doc.size(); i++)
		{
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
			normFre.add((double)wordF.get(diff.get(i))/doc.size());
		}
		
		return normFre;
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
	
	
	public void testWordDistance(String path) throws Exception
	{
		readVector();
		
		ArrayList<String> textArr = readFile(path);
		
		double maxSim = 0;
		
		ArrayList<String> strArr = new ArrayList<String>();
		
		for(int i=0; i<textArr.size()-1; i++)
		{
			if(!wordMap.containsKey(textArr.get(i)))
				continue;
			
		
			if(!wordMap.containsKey(textArr.get(i+1)))
				continue;
				
				
			System.out.print(textArr.get(i) + " " + textArr.get(i+1) + " ");
				
			double d = wordDist(wordMap.get(textArr.get(i)),wordMap.get(textArr.get(i+1)));
				
			
			System.out.println(d);
				
			
		}
		
		/*FileWriter fwS = new FileWriter("C:/Users/jipeng/Desktop/Qiang/dataset/BBC/005.txt");
		BufferedWriter bwS = new BufferedWriter(fwS);
		
		System.out.println(maxSim);
		
		for(int i=0; i<strArr.size()-1; i++)
		{
			
			for(int j=i+1; j<strArr.size(); j++)
			{
				//System.out.print(textArr.get(i) + " " + textArr.get(j) + " ");
				
				double d = wordDist(wordMap.get(strArr.get(i)),wordMap.get(strArr.get(j)));
				
				bwS.write(strArr.get(i) + " " + strArr.get(j) + " " + String.valueOf(maxSim-d));
				bwS.newLine();
				//System.out.println(d);
				
			}
		}
		
		bwS.close();
		fwS.close();*/
		
	}
	
	
	public void testVector()
	{
		readVector();
		
		String s1 = "obama";
		String s2 = "secretary";
		String s3 = "president";
		String s4 = "america";
		String s5 = "government";
		String s6 = "operating";
		String s7 = "system";
		String s8 = "device";
		String s9 = "perform";
		
		System.out.println(s1 + " " + s2 + " " + TermVector.ComputeCosineSimilarity(wordMap.get(s1),wordMap.get(s2)));
		System.out.println(s1 + " " + s3 + " " + TermVector.ComputeCosineSimilarity(wordMap.get(s1),wordMap.get(s3)));
		System.out.println(s2 + " " + s3 + " " + TermVector.ComputeCosineSimilarity(wordMap.get(s2),wordMap.get(s3)));
		System.out.println(s1 + " " + s4 + " " + TermVector.ComputeCosineSimilarity(wordMap.get(s1),wordMap.get(s4)));
		System.out.println(s2 + " " + s4 + " " + TermVector.ComputeCosineSimilarity(wordMap.get(s2),wordMap.get(s4)));
		System.out.println(s1 + " " + s5 + " " + TermVector.ComputeCosineSimilarity(wordMap.get(s1),wordMap.get(s5)));
		System.out.println(s2 + " " + s5 + " " + TermVector.ComputeCosineSimilarity(wordMap.get(s2),wordMap.get(s5)));
		//System.out.println("-----------");
		//System.out.println(s6 + " " + s7 + " " + wordDist(wordMap.get(s6),wordMap.get(s7)));
		//System.out.println(s6 + " " + s8 + " " + wordDist(wordMap.get(s6),wordMap.get(s8)));
		//System.out.println(s8 + " " + s9 + " " + wordDist(wordMap.get(s8),wordMap.get(s9)));
		//System.out.println("-----------");
		//System.out.println(s1 + " " + s8 + " " + wordDist(wordMap.get(s1),wordMap.get(s8)));
		//System.out.println(s2 + " " + s6 + " " + wordDist(wordMap.get(s2),wordMap.get(s6)));
		//System.out.println(s3 + " " + s7 + " " + wordDist(wordMap.get(s3),wordMap.get(s7)));
		
		
		System.out.println("-----------");
		
		TreeSet<WordEntry> closeWords2 = distance("obama");
		
		//Iterator<WordEntry> it2 = closeWords.iterator();
		
		while(!closeWords2.isEmpty())
		{
			WordEntry we = closeWords2.pollFirst();
			System.out.println(we.name + " " + we.score);
		}
		System.out.println("-----------");
		
		System.out.println("-----------");
		
		TreeSet<WordEntry> closeWords3 = distance("china");
		
		//Iterator<WordEntry> it2 = closeWords.iterator();
		
		while(!closeWords3.isEmpty())
		{
			WordEntry we = closeWords3.pollFirst();
			System.out.println(we.name + " " + we.score);
		}
		System.out.println("-----------");
		
		
	}
	
	public void printVector()
	{
		readVector();
		
		String s1 = "annie";
		float []v1 = wordMap.get(s1);
		
		for(int i=0; i<v1.length; i++)
			System.out.print(v1[i]+ " ");
		System.out.println();
		String s2 = "relationship";
		float []v2 = wordMap.get(s2);
		for(int i=0; i<v2.length; i++)
			System.out.print(v2[i]+ " ");
		System.out.println();
		TermVector.ComputeCosineSimilarity(v1,v2);
		for(int i=0; i<v2.length; i++)
			System.out.print(v2[i]+ " ");
		System.out.println();
		//System.out.println(s6 + " " + s8 + " " + wordDist(wordMap.get(s6),wordMap.get(s8)));
		//System.out.println(s8 + " " + s9 + " " + wordDist(wordMap.get(s8),wordMap.get(s9)));
		//System.out.println("-----------");
		//System.out.println(s1 + " " + s8 + " " + wordDist(wordMap.get(s1),wordMap.get(s8)));
		//System.out.println(s2 + " " + s6 + " " + wordDist(wordMap.get(s2),wordMap.get(s6)));
		//System.out.println(s3 + " " + s7 + " " + wordDist(wordMap.get(s3),wordMap.get(s7)));
		
		
		
		
		
	}
	
	public double compuDiver2(ArrayList<String> doc)
	{
		//System.out.println(doc.size());
		
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
				dist += d*normFre.get(j);
				//num++;
			}
			//if(num>0)
				//dist /= num;
			
			divS += dist*normFre.get(i);
		}
		
		//divS += divS;
		
		//divS /= diff.size()*(diff.size()-1); 
		
		return divS;//*Math.log(doc.size());
	}
	
	public double compuDiver2(String path1,String path2)
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
	
	public Map<String,Map<String,Double>> readGraph(ArrayList<String> diff) 
	{
		Map<String,Map<String,Double>> result = new HashMap<String,Map<String,Double>>();
		try {
			for(int i=0; i<diff.size(); i++)
			{
				//int index1 = allWord.indexOf(diff.get(i));
				
				if(!wordMap.containsKey(diff.get(i)))
					continue;
				
				String source = diff.get(i);
				for(int j=0; j<diff.size(); j++)
				{
					if(j==i)
						continue;
					//int index2 = allWord.indexOf(diff.get(j));
					
					if(!wordMap.containsKey(diff.get(j)))
						continue;
					
					String target = diff.get(j);
					//System.out.print(diff.get(i) + " " + diff.get(j) + " ");
					
					double sis = wordSisCosine(wordMap.get(source),wordMap.get(target));
					
					//double sis = 10-dist;
					if(sis<0)
						sis = 0;
					if (result.get(source) == null) result.put(source, new HashMap<String,Double>());
					result.get(source).put(target, sis);
				}
			}
		} catch (Exception e) {
		      System.err.println("Exception while reading the graph:"); 
			  System.err.println(e);
			  System.exit(1);
		}
		return result;
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
	
	/**
	 * Returns a symmetric version of the given graph.
	 * A graph is symmetric if and only if for each pair of nodes,
	 * the weight of the edge from the first to the second node
	 * equals the weight of the edge from the second to the first node.
	 * Here the symmetric version is obtained by adding to each edge weight
	 * the weight of the inverse edge.
	 * 
	 * @param graph  possibly unsymmetric graph.
	 * @return symmetric version of the given graph.
	 */
	private static Map<String,Map<String,Double>> makeSymmetricGraph
			(Map<String,Map<String,Double>> graph) {
		Map<String,Map<String,Double>> result = new HashMap<String,Map<String,Double>>();
		for (String source : graph.keySet()) {
			for (String target : graph.get(source).keySet()) {
				double weight = graph.get(source).get(target);
				double revWeight = 0.0f;
				if (graph.get(target) != null && graph.get(target).get(source) != null) {
					revWeight = graph.get(target).get(source);
				}
				if (result.get(source) == null) result.put(source, new HashMap<String,Double>());
				result.get(source).put(target, weight+revWeight);
				if (result.get(target) == null) result.put(target, new HashMap<String,Double>());
				result.get(target).put(source, weight+revWeight);
			}
		}
		return result;
	}
	
	/**
	 * Construct a map from node names to nodes for a given graph, 
	 * where the weight of each node is set to its degree,
     * i.e. the total weight of its edges. 
	 * 
	 * @param graph the graph.
	 * @return map from each node names to nodes.
	 */
	private static Map<String,Node> makeNodes(Map<String,Map<String,Double>> graph) {
		Map<String,Node> result = new HashMap<String,Node>();
		for (String nodeName : graph.keySet()) {
            double nodeWeight = 0.0;
            for (double edgeWeight : graph.get(nodeName).values()) {
                nodeWeight += edgeWeight;
            }
			result.put(nodeName, new Node(nodeName, nodeWeight));
		}
		return result;
	}
	
	/**
     * Converts a given graph into a list of edges.
     * 
     * @param graph the graph.
     * @param nameToNode map from node names to nodes.
     * @return the given graph as list of edges.
     */
    private static List<Edge> makeEdges(Map<String,Map<String,Double>> graph, 
            Map<String,Node> nameToNode) {
        List<Edge> result = new ArrayList<Edge>();
        for (String sourceName : graph.keySet()) {
            for (String targetName : graph.get(sourceName).keySet()) {
                Node sourceNode = nameToNode.get(sourceName);
                Node targetNode = nameToNode.get(targetName);
                double weight = graph.get(sourceName).get(targetName);
                result.add( new Edge(sourceNode, targetNode, weight) );
            }
        }
        return result;
    }
    
    /**
	 * Returns, for each node in a given list,
	 * a random initial position in two- or three-dimensional space. 
	 * 
	 * @param nodes node list.
     * @param is3d initialize 3 (instead of 2) dimension with random numbers.
	 * @return map from each node to a random initial positions.
	 */
	private static Map<Node,double[]> makeInitialPositions(List<Node> nodes, boolean is3d) {
        Map<Node,double[]> result = new HashMap<Node,double[]>();
		for (Node node : nodes) {
            double[] position = { Math.random() - 0.5,
                                  Math.random() - 0.5,
                                  is3d ? Math.random() - 0.5 : 0.0 };
            result.put(node, position);
		}
		return result;
	}
	
	// compute the score of diversity based on graph clustering
	public double compuDiverBasedClustering(String path1,String path2)
	{
		double divS = 0.0;
		ArrayList<String> doc1 = readFile(path1);
		ArrayList<String> doc2 = readFile(path2);
		ArrayList<String> doc = new ArrayList<String>();
		
		doc.addAll(doc1);
		doc.addAll(doc2);
		
		ArrayList<String> diff = new ArrayList<String>();
		
		ArrayList<Integer> textWordFre = computWordFre(doc, diff);
		
		Map<String,Map<String,Double>> graph = readGraph(diff);
		
		graph = makeSymmetricGraph(graph);
		Map<String,Node> nameToNode = makeNodes(graph);
        List<Node> nodes = new ArrayList<Node>(nameToNode.values());
        List<Edge> edges = makeEdges(graph,nameToNode);
		Map<Node,double[]> nodeToPosition = makeInitialPositions(nodes, true);
		// see class MinimizerBarnesHut for a description of the parameters;
		// for classical "nice" layout (uniformly distributed nodes), use
		//new MinimizerBarnesHut(nodes, edges, -1.0, 2.0, 0.05).minimizeEnergy(nodeToPosition, 100);
		new MinimizerBarnesHut(nodes, edges, 0.0, 1.0, 0.05).minimizeEnergy(nodeToPosition, 100);
        // see class OptimizerModularity for a description of the parameters
        Map<Node,Integer> nodeToCluster = 
            new OptimizerModularity().execute(nodes, edges, false);
		
        ArrayList<ArrayList<String>> wordCluster = getWordCluster(nodeToCluster);
        
        divS = diverMeasure(wordCluster,diff,textWordFre);
		//divS += divS;
		
		//divS /= diff.size()*(diff.size()-1); 
		//System.out.println("the score of diversity: " + divS);
		return divS;
	}
	
	public void printTextVector(String path1,String path2) throws Exception
	{
	
		ArrayList<String> doc1 = readFile(path1);
		ArrayList<String> doc2 = readFile(path2);
		ArrayList<String> doc = new ArrayList<String>();
		
		doc.addAll(doc1);
		doc.addAll(doc2);
		
		ArrayList<String> diff = new ArrayList<String>();
		
		ArrayList<Integer> textWordFre = computWordFre(doc, diff);
		
		
		FileWriter pFW = new FileWriter("wordForVS2.txt");
		BufferedWriter wordBW = new BufferedWriter(pFW);
		
		FileWriter tFW = new FileWriter("textVectorS2.txt");
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
	}
	
	public double diverMeasure(ArrayList<ArrayList<String>> wordCluster, ArrayList<String> diffWord, ArrayList<Integer> textWordFre)
	{
		
		ArrayList<ArrayList<Double>> feat = new ArrayList<ArrayList<Double>>();
		
		Vector<Integer> numOfWords = new Vector<Integer>();
		
		int totalW = 0;
		
		for(int i=0; i<textWordFre.size(); i++)
			totalW += textWordFre.get(i);
		
		for(int i=0; i<wordCluster.size(); i++)
		{
			ArrayList<String> oneClus = wordCluster.get(i);
			
			ArrayList<Double> oneF = new ArrayList<Double>();
			
			int tot = 0;
			for(int j=0; j<diffWord.size(); j++)
			{
				if(oneClus.contains(diffWord.get(j)))
				{
					int fre = textWordFre.get(j);
					tot += fre;
					oneF.add((double)fre);
				}
				//else
				//	oneF.add(0.0);
			}
			
			
			for(int j=0; j<oneF.size(); j++)
				oneF.set(j, oneF.get(j)/tot);
			feat.add(oneF);
			
			numOfWords.add(tot);
		}
		
		double divS = 0;
		//System.out.println("the number of cluster: " + wordCluster.size());
		for(int i=0; i<wordCluster.size(); i++)
		{
			//System.out.println("the number of words: " + numOfWords.get(i));
			
			//System.out.println("i: " + i);
			for(int j=0; j<wordCluster.size(); j++)
			{
				//System.out.println("j: " + j + " word number : " + wordCluster.get(j));
				if(i==j)
				{
					divS += ((double)numOfWords.get(i)/totalW)*((double)numOfWords.get(j)/totalW);
				}else
				{
					
					ArrayList<Double> distWTW = new ArrayList<Double>();
					
					for(int ii=0; ii<wordCluster.get(i).size(); ii++)
					{
						//ArrayList<Double> oneV = new ArrayList<Double>();
						
						//String w = diffWord.get(ii);
						String w = wordCluster.get(i).get(ii);
						
						for(int jj=0; jj<wordCluster.get(j).size(); jj++)
						{
							String wj = wordCluster.get(j).get(jj);
							
							double dist = wordDist(wordMap.get(w), wordMap.get(wj));
								
							distWTW.add(dist);
						}
					//	distWTW.add(oneV);
						
						/*if(!wordCluster.get(i).contains(w) && !wordCluster.get(j).contains(w))
						{
							continue;
						}else if(!wordCluster.get(i).contains(w))
						{
							for(int jj=0; jj<feat.get(i).size(); jj++)
								oneV.add(0.0);
							distWTW.add(oneV);
						}else
						{
							for(int jj=0; jj<diffWord.size(); jj++)
							{
								String wj = diffWord.get(jj);
								
								if(!wordCluster.get(i).contains(wj) && !wordCluster.get(j).contains(wj))
								{
									continue;
								}
								
								if(!wordCluster.get(j).contains(wj))
								{
									oneV.add(0.0);
								}else
								{
									double dist = wordDist(wordMap.get(w), wordMap.get(wj));
									
									oneV.add(Double.parseDouble(df.format(dist)));
								}
								
							}
							distWTW.add(oneV);
						}*/
					}
					//JFastEMD emd = new JFastEMD();
					
					//printArr(feat.get(i),feat.get(j),distWTW);
					
					//double sis = emd.emdHat(indI, indJ, distWTW, 0.0);
					double sis = docDist(feat.get(i),feat.get(j),distWTW);
					//System.out.println("the " + i + " cluster" + " and the " + j + " cluster:" + sis);
					/*ArrayList<Double> distWTW2 = new ArrayList<Double>();
					for(int tt=0; tt<feat.get(i).size(); tt++)
						for(int gg=0; gg<feat.get(i).size(); gg++)
							distWTW2.add(0.0);
					double sis2 = docSis(feat.get(i),feat.get(i),distWTW);
					System.out.println("the " + i + " cluster" + " and the " + i + " cluster:" + sis2);*/
					/*if(sis<0.1)
					{
						printArr(indI,indJ,distWTW);
					}*/
					divS += ((double)numOfWords.get(i)/totalW)*((double)numOfWords.get(j)/totalW)*(sis);
				}
			}
		}
		return divS;
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
	
	
	public void printArr(Vector<Double> P, Vector<Double> Q, Vector<Vector<Double>> C)
	{
		try
		{
			FileWriter pFW = new FileWriter("P.txt");
			BufferedWriter pBW = new BufferedWriter(pFW);
			
			for(int i=0; i<P.size(); i++)
			{
				pBW.write(String.valueOf(P.get(i))+ " ");
			}
			
			pBW.close();
			
			FileWriter qFW = new FileWriter("Q.txt");
			BufferedWriter qBW = new BufferedWriter(qFW);
			
			for(int i=0; i<Q.size(); i++)
			{
				qBW.write(String.valueOf(Q.get(i))+ " ");
			}
			
			qBW.close();
			
			FileWriter cFW = new FileWriter("C.txt");
			BufferedWriter cBW = new BufferedWriter(cFW);
			
			for(int i=0; i<C.size(); i++)
			{
				Vector<Double> oneV = C.get(i);
				for(int j=0; j<oneV.size(); j++)
					cBW.write(String.valueOf(oneV.get(j))+ " ");
				cBW.newLine();
			}
			cBW.close();
			
		}catch(Exception e)
		{
			e.printStackTrace();
		}
	
		
	}
	
	
	public ArrayList<ArrayList<String>> getWordCluster(Map<Node,Integer> nodeToCluster)
	{
		ArrayList<ArrayList<String>> wordCluster = new ArrayList<ArrayList<String>>();
		
		Iterator<Node> it = nodeToCluster.keySet().iterator();
		
		ArrayList<Integer> diff = new ArrayList<Integer>();
		while(it.hasNext())
		{
			Node n = it.next();
			
			String word = n.name;
			
			int clusterId = nodeToCluster.get(n);
			
			if(!diff.contains(clusterId))
			{
				ArrayList<String> clu = new ArrayList<String>();
				clu.add(word);
				wordCluster.add(clu);
				diff.add(clusterId);
			}else
			{
				wordCluster.get(diff.indexOf(clusterId)).add(word);
			}
			
		}
		
		return wordCluster;
	}
	
	public void test()
	{
		ArrayList<String> doc = new ArrayList<String>();
		
		readVector();
		/*Set<WordEntry> closeWords = distance("frog");
		
		Iterator<WordEntry> it = closeWords.iterator();
		
		while(it.hasNext())
		{
			WordEntry we = it.next();
			System.out.println(we.name + " " + we.score);
		}*/
		System.out.println("-----------");
		
		TreeSet<WordEntry> closeWords2 = distance("crane");
		
		//Iterator<WordEntry> it2 = closeWords.iterator();
		
		while(!closeWords2.isEmpty())
		{
			WordEntry we = closeWords2.pollFirst();
			System.out.println(we.name + " " + we.score);
		}
		System.out.println("-----------");
		String s1 = "crane";
		String s2 = "bird";
		String s3 = "machine";
		
		//String s4 = "account";
		//String s5 = "tree";
		
		System.out.println(s1 +" " + s2+ wordDist(wordMap.get(s1),wordMap.get(s2)));
		System.out.println(s1  +" " + s3+ wordDist(wordMap.get(s1),wordMap.get(s3)));
		System.out.println(s2  +" " + s3+ wordDist(wordMap.get(s2),wordMap.get(s3)));
		//System.out.println(s1  +" " + s5+ wordDist(wordMap.get(s1),wordMap.get(s5)));
		System.out.println("-----------");
		
		TreeSet<WordEntry> closeWords = distance("apple");
		
		//Iterator<WordEntry> it2 = closeWords.iterator();
		
		while(!closeWords.isEmpty())
		{
			WordEntry we = closeWords.pollFirst();
			System.out.println(we.name + " " + we.score);
		}
		System.out.println("-----------");
		//doc.add(s1);
		//doc.add(s2);
		//doc.add(s3);
		//doc.add(s7);
		//doc.add(s8);
		//System.out.println(" aaaa");
		System.out.println(wordDist(wordMap.get("nba"),wordMap.get("basketball")));
		System.out.println(wordDist(wordMap.get("nba"),wordMap.get("football")));
		System.out.println(wordDist(wordMap.get("dad"),wordMap.get("father")));
		System.out.println(wordDist(wordMap.get("books"),wordMap.get("book")));
		System.out.println(wordDist(wordMap.get("nba"),wordMap.get("lebron")));
		System.out.println(wordDist(wordMap.get("sport"),wordMap.get("art")));
		/*System.out.println("aaaaa");
		
		System.out.println(compuDiver2(doc));
		ArrayList<String> doc2 = new ArrayList<String>();
		doc2.add(s1);
		//doc2.add(s2);
		doc2.add(s3);
		doc2.add(s7);
		doc2.add(s4);
		
		System.out.println(compuDiver2(doc2));
		ArrayList<String> doc3 = new ArrayList<String>();
		doc3.add(s1);
		doc3.add(s3);
		doc3.add(s7);
		doc3.add(s9);
		System.out.println(compuDiver2(doc3));*/
		
		String t1 = "airbus";
		if(wordMap.containsKey(t1))
			System.out.println(t1);
		else
			System.out.println("no");
		
		String t2 = "a380";
		if(wordMap.containsKey(t2))
			System.out.println(t2);
		else
			System.out.println("no");
		
		String t3 = "antarctic";
		if(wordMap.containsKey(t3))
			System.out.println(t3);
		else
			System.out.println("no");
		
		
	}
	
	public TreeSet<WordEntry> distance(String queryWord) {

		float[] center = wordMap.get(queryWord);
		if (center == null) {
			return null;
		}

		int resultSize = wordMap.size() < topNSize ? wordMap.size() : topNSize;
		TreeSet<WordEntry> result = new TreeSet<WordEntry>();

		double min = Float.MIN_VALUE;
		for (Map.Entry<String, float[]> entry : wordMap.entrySet()) {
			float[] vector = entry.getValue();
			
			float dist = TermVector.computDist(center,vector);

			//if (dist > 0) {
				result.add(new WordEntry(entry.getKey(), dist));
				if (resultSize < result.size()) {
					result.pollLast();
				}
				//min = result.first().score;
			//}
		}
		result.pollFirst();

		return result;
	}

	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		try
		{
			EmbeddingDiver ed = new EmbeddingDiver();
			
			//ed.readVector();
			
			ed.testVector();
			//ed.readFile("C:/Users/jipeng/Desktop/dataset/BBC/bbc/business/001.txt");
			//ed.wordDist("", w2)
			//ed.testWordDistance("C:/Users/jipeng/Desktop/dataset/BBC/005.txt");
			//ed.testVector();
			//ed.printVector();
			//ed.compuDiverBasedClustering("C:/Users/jipeng/Desktop/dataset/001.txt", "C:/Users/jipeng/Desktop/dataset/002.txt");
			//ed.printTextVector("C:/Users/jipeng/Desktop/dataset/BBC/bbc/business/002.txt", "C:/Users/jipeng/Desktop/dataset/BBC/bbc/business/003.txt");
			//ed.printTextVector("C:/Users/jipeng/Desktop/dataset/BBC/bbc/business/002.txt", "C:/Users/jipeng/Desktop/dataset/BBC/bbc/entertainment/001.txt");
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		
	}

}
