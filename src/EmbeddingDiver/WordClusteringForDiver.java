package EmbeddingDiver;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import textcluster.TermVector;
import textcluster.WawaCluster;
import textcluster.WawaKMeans;
import kex.stopwords.Stopwords;
import kex.stopwords.StopwordsEnglish;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;



public class WordClusteringForDiver {

	HashMap<String,Integer> wordIdMap = new HashMap<String,Integer>();
	
	
	String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/glove.6B.300d.txt";
	//String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/vectors.840B.300d.txt";
	//String vectorPath = "C:/Users/jipeng/Desktop/Qiang/Word2Vec/deps.words";
	//private int topNSize = 40;
	//float vect[][] = new float[400000][300];
	
	//ArrayList<String> allWord = new ArrayList<String>();
	public HashMap<String, float[]> wordMap = new HashMap<String, float[]>();
	
	int wordDemon = -1;
	
	//MaxentTagger tagger = new MaxentTagger("C:/Users/jipeng/Desktop/TopicModel/stanford-postagger-2014-01-04/models/english-left3words-distsim.tagger");
	
	//private Stopwords m_EnStopwords = new StopwordsEnglish();
	
	//Pattern p = Pattern.compile("[^a-zA-Z]", Pattern.CASE_INSENSITIVE);
	
	//DecimalFormat df = new DecimalFormat("0.0000");
	double bigThres;
	double smallThres;
	
	public WordClusteringForDiver()
	{
		smallThres = 0.2;
		bigThres = 0.35;
	}
	
	public void initialPara(double bigThres, double smallThres)
	{
		this.bigThres = bigThres;
		this.smallThres = smallThres;
	}
	
	public WordClusteringForDiver(double bigThres, double smallThres)
	{
		this.bigThres = bigThres;
		this.smallThres = smallThres;
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
				wordDemon = word.length-1;
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
	
	public double computTwoArr(ArrayList<String> arr1, ArrayList<String> arr2)
	{
		double sis = .0;
		
		for(int i=0; i<arr1.size(); i++)
		{
			for(int j=0; j<arr2.size(); j++)
			{
				sis += (double)TermVector.ComputeCosineSimilarity(wordMap.get(arr1.get(i)),wordMap.get(arr2.get(j)));
			}
		}
		
		return sis/(arr1.size()*arr2.size());
	}
	
	public void computDiffSis(ArrayList<String> diff, ArrayList<Double> sis, ArrayList<Integer> indArr)
	{
		for(int i=0; i<diff.size(); i++)
			for(int j=i+1; j<diff.size(); j++)
			{
				double s = (double)TermVector.ComputeCosineSimilarity(wordMap.get(diff.get(i)),wordMap.get(diff.get(j)));
				
				if(s>=bigThres)
				{
					sis.add(s);
					indArr.add(i*10000+j);
				}
			}
	}
	
	public void sortSis(ArrayList<Double> sis, ArrayList<Integer> indArr)
	{
		for(int i=0; i<sis.size(); i++)
		{
			double cur = sis.get(i);
			
			int curInd = indArr.get(i);
			
			double maxV = sis.get(i);
			int maxInd = i;
			int ind = -1;
			for(int j=i+1; j<sis.size(); j++)
			{
				if(sis.get(j)>maxV)
				{
					maxV = sis.get(j);
					maxInd = j;
					ind = indArr.get(j);
				}
			}
			if(i!=maxInd)
			{
				
				sis.set(i, maxV);
				sis.set(maxInd, cur);
				
				indArr.set(i, ind);
				indArr.set(maxInd, curInd);
			
			}
			//indArr.add(minInd);
		}
	}
	
	public void combineCluster(ArrayList<ArrayList<String>> clustArr)
	{
		double maxSis = 0.0;
		do{
			maxSis = 0.0;
			int smallInd = -1;
			int bigInd = -1;
			
			for(int i=0; i<clustArr.size()-1; i++)
			{
				for(int j=i+1; j<clustArr.size(); j++)
				{
					double sisT = computTwoArr(clustArr.get(i),clustArr.get(j));
					if(sisT>=smallThres)
					{
						if(sisT>maxSis)
						{
							maxSis = sisT;
							smallInd = i;
							bigInd = j;
						}
						
					}
				}
			}
			
			if(maxSis>smallThres)
			{
				clustArr.get(smallInd).addAll(clustArr.get(bigInd));
				clustArr.remove(bigInd);
			}
			
		}while(maxSis>smallThres);
	}
	
	public ArrayList<ArrayList<String>> computCluster( ArrayList<String> diff, ArrayList<Double> sis, ArrayList<Integer> indArr)
	{
		ArrayList<ArrayList<String>> clustArr = new ArrayList<ArrayList<String>>();
		int []cluInd = new int[diff.size()];
		
		for(int i=0; i<cluInd.length; i++)
			cluInd[i] = -1;
		for(int i=0; i<sis.size(); i++)
		{
			double wei = sis.get(i);
			int ind = indArr.get(i);
			int rInd = ind/10000;
			int cInd = ind%10000;
			
			//if(wei>=sisThreshold)
			//{
				int cluIdR = cluInd[rInd];
				int cluIdC = cluInd[cInd];
				
				if(cluIdR==-1 && cluIdC==-1)
				{
					ArrayList<String> clu = new ArrayList<String>();
					clu.add(diff.get(rInd));
					clu.add(diff.get(cInd));
					cluInd[rInd]=clustArr.size();
					cluInd[cInd]=clustArr.size();
					clustArr.add(clu);
				}
				else if(cluIdR==-1)
				{
					clustArr.get(cluIdC).add(diff.get(rInd));
					cluInd[rInd] = cluIdC;
				}else if(cluIdC==-1)
				{
					clustArr.get(cluIdR).add(diff.get(cInd));
					cluInd[cInd] = cluIdR;
				}else if(cluIdC==cluIdR)
				{
					continue;
				}else if(cluIdC!=cluIdR)
				{
					ArrayList<String> strR = clustArr.get(cluIdR);
					ArrayList<String> strC = clustArr.get(cluIdC);
					
					double twoSis = computTwoArr(strR,strC);
					
					if(twoSis>smallThres)
					{
						if(cluIdC<cluIdR)
						{
							for(int j=0; j<cluInd.length; j++)
							{
								if(cluInd[j] ==  cluIdR)
									cluInd[j] = cluIdC;
								if(cluInd[j]>cluIdR)
									cluInd[j] -= 1;
							}
							
							strC.addAll(strR);
							
							clustArr.remove(cluIdR);
						}else
						{
							for(int j=0; j<cluInd.length; j++)
							{
								if(cluInd[j] ==  cluIdC)
									cluInd[j] = cluIdR;
								if(cluInd[j]>cluIdC)
									cluInd[j] -= 1;
							}
							
							strR.addAll(strC);
							
							clustArr.remove(cluIdC);
						}
					}
				}else
				{
					System.out.println("Wrong!");
				}
			//}
			//sis.remove(0);
			//indArr.remove(0);
		}
		
		ArrayList<String> existArr = new ArrayList<String>();
		
		for(int i=0; i<clustArr.size(); i++)
			existArr.addAll(clustArr.get(i));
		
		for(int i=0; i<diff.size(); i++)
			if(!existArr.contains(diff.get(i)))
			{
				ArrayList<String> oneA = new ArrayList<String>();
				oneA.add(diff.get(i));
				clustArr.add(oneA);
			}
		combineCluster(clustArr);
		return clustArr;
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
			
			for(int j=0; j<oneClus.size(); j++)
			{
				String word =  oneClus.get(j);
				int fre = textWordFre.get(diffWord.indexOf(word));
				tot += fre;
				oneF.add((double)fre);
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
			//System.out.println("the " + i + " number of words: " + numOfWords.get(i));
			
			//System.out.print("i:" + i +" ");
			
			
			for(int j=0; j<wordCluster.size(); j++)
			{
				//System.out.println("j: " + j + " word number : " + numOfWords.get(j));
				if(i==j)
				{
					continue;
					
					//divS += ((double)numOfWords.get(i)/totalW)*((double)numOfWords.get(i)/totalW)*distK[i][j];
					
				}else
				{
					
					ArrayList<Double> distWTW = new ArrayList<Double>();
					
					for(int ii=0; ii<wordCluster.get(i).size(); ii++)
					{
						String w = wordCluster.get(i).get(ii);
						
						for(int jj=0; jj<wordCluster.get(j).size(); jj++)
						{
							String wj = wordCluster.get(j).get(jj);
							
							//double dist = (1-TermVector.ComputeCosineSimilarity(wordMap.get(w), wordMap.get(wj)));
							double dist = TermVector.computDist(wordMap.get(w), wordMap.get(wj));	
							distWTW.add(dist);
						}
			
					}
					
					double dist = docDist(feat.get(i),feat.get(j),distWTW);
					//dist *= (-(2.3/0.81)*(dist-0.9)*(dist-0.9)+3.0);
					//double dist = Math.exp(docDist(feat.get(i),feat.get(j),distWTW));
					//System.out.print( j + ":"+ dist + " ");
					//double dist = docDist(feat.get(i),feat.get(j),distWTW);
					
					divS += ((double)numOfWords.get(i)/totalW)*((double)numOfWords.get(j)/totalW)*(dist);
				}
			}
			//System.out.println();
		}
		return divS;
	}
	
	public double diverMeasure(ArrayList<ArrayList<String>> wordCluster,ArrayList<String> diffWord, ArrayList<Integer> textWordFre, double [][]dist)
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
			
			for(int j=0; j<oneClus.size(); j++)
			{
				String word =  oneClus.get(j);
				int fre = textWordFre.get(diffWord.indexOf(word));
				tot += fre;
				oneF.add((double)fre);
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
			//System.out.println("the " + i + " number of words: " + numOfWords.get(i));
			
			//System.out.print("i:" + i +" ");
			
			
			for(int j=0; j<wordCluster.size(); j++)
			{
				//System.out.println("j: " + j + " word number : " + numOfWords.get(j));
				if(i==j)
				{
					continue;
					
					//divS += ((double)numOfWords.get(i)/totalW)*((double)numOfWords.get(i)/totalW)*distK[i][j];
					
				}else
				{
					
					
					//System.out.print( j + ":"+ dist + " ");
					//double dist = docDist(feat.get(i),feat.get(j),distWTW);
					
					divS += ((double)numOfWords.get(i)/totalW)*((double)numOfWords.get(j)/totalW)*(dist[i][j]);
				}
			}
			//System.out.println();
		}
		return divS;
	}
	
	public double computDiverForOneTopic(ArrayList<String> doc)
	{
		double dist = .0;
		
		for(int i=0; i<doc.size()-1; i++)
		{
			for(int j=i+1; j<doc.size(); j++)
			{
				dist += (double)TermVector.computDist(wordMap.get(doc.get(i)),wordMap.get(doc.get(j)));
			}
		}
		
		return 2*dist/(doc.size()*(doc.size()-1));
	}
	
	public ArrayList<ArrayList<String>> kmeansCluster(ArrayList<String> diff, int K)
	{
		float [][]data = new float[diff.size()][wordDemon];
		
		for(int i=0; i<diff.size(); i++)
		{
			float vec[] = wordMap.get(diff.get(i));
			for(int j=0; j<wordDemon; j++)
				data[i][j] =vec[j];
		}
		
		WawaKMeans kmeans = new WawaKMeans(data, K);
        
		/*String s2 = "relationship";
		float []v2 = wordMap.get(s2);
		System.out.print("After initial functtion!");
		for(int ii=0; ii<v2.length; ii++)
			System.out.print(v2[ii]+ " ");
		System.out.println();
		*/
        kmeans.Start();

       // String s2 = "relationship";
		/*float []v3 = wordMap.get(s2);
		System.out.print("After start functtion!");
		for(int ii=0; ii<v3.length; ii++)
			System.out.print(v3[ii]+ " ");
		System.out.println();*/
         //kmeans.getDistForClusters(distKmeans);
        //6����ȡ���������
        WawaCluster[] clusters = kmeans.getClusters();
       // kmeans = null;
        ArrayList<ArrayList<String>> resClus = new ArrayList<ArrayList<String>>();
        for(WawaCluster cluster : clusters){

            List<Integer> members = cluster.CurrentMembership;
            
            ArrayList<String> oneT = new ArrayList<String>();
           // System.out.println("-----------------");
            for (int i =0; i<members.size(); i++)
            {
            	oneT.add(diff.get(members.get(i)));
            	//System.out.print(diff.get(members.get(i))+ " ");
            }
            resClus.add(oneT);
           // System.out.println();
        
        }
		return resClus;
	}
	
	public ArrayList<ArrayList<String>> kmeansCluster(ArrayList<String> diff, int K, double [][]distKmeans)
	{
		float [][]data = new float[diff.size()][wordDemon];
		
		for(int i=0; i<diff.size(); i++)
		{
			float vec[] = wordMap.get(diff.get(i));
			for(int j=0; j<wordDemon; j++)
				data[i][j] =vec[j];
		}
		
		WawaKMeans kmeans = new WawaKMeans(data, K);
        
		/*String s2 = "relationship";
		float []v2 = wordMap.get(s2);
		System.out.print("After initial functtion!");
		for(int ii=0; ii<v2.length; ii++)
			System.out.print(v2[ii]+ " ");
		System.out.println();
		*/
        kmeans.Start();

       // String s2 = "relationship";
		/*float []v3 = wordMap.get(s2);
		System.out.print("After start functtion!");
		for(int ii=0; ii<v3.length; ii++)
			System.out.print(v3[ii]+ " ");
		System.out.println();*/
         kmeans.getDistForClusters(distKmeans);
        //6����ȡ���������
        WawaCluster[] clusters = kmeans.getClusters();
       // kmeans = null;
        ArrayList<ArrayList<String>> resClus = new ArrayList<ArrayList<String>>();
        for(WawaCluster cluster : clusters){

            List<Integer> members = cluster.CurrentMembership;
            
            ArrayList<String> oneT = new ArrayList<String>();
           // System.out.println("-----------------");
            for (int i =0; i<members.size(); i++)
            {
            	oneT.add(diff.get(members.get(i)));
            	//System.out.print(diff.get(members.get(i))+ " ");
            }
            resClus.add(oneT);
           // System.out.println();
        
        }
		return resClus;
	}
	
	
	public ArrayList<String> removeHardWords(ArrayList<String> doc)
	{
		ArrayList<String> remainWords = new ArrayList<String>();
		//System.out.println("********************");
		for(int i=0; i<doc.size(); i++)
		{
			if(!wordMap.containsKey(doc.get(i)))
				continue;
			remainWords.add(doc.get(i));
		}
		ArrayList<String> remainWords2 = new ArrayList<String>();
		for(int i=0; i<remainWords.size(); i++)
		{
			double allDist = 0D;
			
			//int num=0;
			
			int flag = 0;
			for(int j=0; j<remainWords.size(); j++)
			{
				if(j!=i)
				{
					double  dist2 = TermVector.computDist(wordMap.get(remainWords.get(i)),wordMap.get(remainWords.get(j)));
					if(dist2==0)
						continue;
					
					if(dist2<=0.85)
						flag++;
					//if(flag>=2)
						//break;
					//if(dist2>1.0)
						//num++;
						//System.out.println(dist2 + " " + remainWords.get(i) + " " + remainWords.get(j) );
					allDist += dist2;
				}
			}
			if(flag >=2 || allDist<=0.9*(remainWords.size()-1))
			//if(num<(remainWords.size()-5))
				remainWords2.add(remainWords.get(i));
			else
			{
				//System.out.print(remainWords.get(i)+ " ");
				/*for(int j=0; j<remainWords.size(); j++)
				{
					if(j!=i)
					{
						double  dist2 = wordDist(wordMap.get(remainWords.get(i)),wordMap.get(remainWords.get(j)));
						//if(dist2>1.0)
							//num++;
						System.out.println(dist2 + " " + remainWords.get(i) + " " + remainWords.get(j) );
						//allDist += dist2;
					}
				}*/
			}
				
		}
		//System.out.println("");
		return remainWords2;
	}
	
	public double compuDiverBasedKmeans(ArrayList<String> doc)
	{
		double divS = 0.0;
		
		ArrayList<String> rdoc = removeHardWords(doc);
		
		/*String s2 = "relationship";
		float []v2 = wordMap.get(s2);
		System.out.print("After remove functtion!");
		for(int ii=0; ii<v2.length; ii++)
			System.out.print(v2[ii]+ " ");
		System.out.println();*/
		
		ArrayList<String> diff = new ArrayList<String>();
		
		ArrayList<Integer> freArr = normBagOfWords(rdoc, diff);
		
		//int docLen = doc.size();
		//double sisThreshold = ;
		
		//ArrayList<Double> sis = new ArrayList<Double>();
		//ArrayList<Integer> indArr = new ArrayList<Integer>();
		
		//computDiffSis(diff,sis,indArr);
		
		//sortSis(sis,indArr);
		int k = 20;
		//double [][]dist = new double[k][k];
		//ArrayList<ArrayList<String>> clustArr = kmeansCluster(diff, k);
		ArrayList<ArrayList<String>> clustArr = kmeansCluster(diff, k);
		/*float []v3 = wordMap.get(s2);
		System.out.print("After kmeans functtion!");
		for(int ii=0; ii<v3.length; ii++)
			System.out.print(v3[ii]+ " ");
		System.out.println();*/
		
		/*System.out.println("----------------" );
		for(int i=0; i<dist.length; i++)
		{
			for(int j=0; j<dist[i].length; j++)
			{
				System.out.print(dist[i][j]+ " ");
			}
			System.out.println();
		}*/
		/*for(int i=0; i<clustArr.size();i++)
			if(clustArr.get(i).size()<2)
			{
				clustArr.remove(i);
				i--;
			}*/
		//for(int i=0; i<clustArr.size();i++)
			//System.out.println(i+ " " + clustArr.get(i).toString());
		
		divS = diverMeasure(clustArr,diff,freArr);
		
	/*	float []v4 = wordMap.get(s2);
		System.out.print("After diverMeasure functtion!");
		for(int ii=0; ii<v4.length; ii++)
			System.out.print(v4[ii]+ " ");
		System.out.println();*/
		return divS;//*Math.log(doc.size());
	}
	
	public double compuDiverBasedKmeansMean(ArrayList<String> doc)
	{
		double divS = 0.0;
		
		ArrayList<String> rdoc = removeHardWords(doc);
		
		
		
		ArrayList<String> diff = new ArrayList<String>();
		
		ArrayList<Integer> freArr = normBagOfWords(rdoc, diff);
		
		int k = 20;
		double [][]dist = new double[k][k];
		//ArrayList<ArrayList<String>> clustArr = kmeansCluster(diff, k);
		ArrayList<ArrayList<String>> clustArr = kmeansCluster(diff, k,dist);
		
		
		/*System.out.println("----------------" );
		for(int i=0; i<dist.length; i++)
		{
			for(int j=0; j<dist[i].length; j++)
			{
				System.out.print(dist[i][j]+ " ");
			}
			System.out.println();
		}*/
		//for(int i=0; i<clustArr.size();i++)
			//System.out.println(i+ " " + clustArr.get(i).toString());
		
		divS = diverMeasure(clustArr,diff,freArr,dist);
		
	/*	float []v4 = wordMap.get(s2);
		System.out.print("After diverMeasure functtion!");
		for(int ii=0; ii<v4.length; ii++)
			System.out.print(v4[ii]+ " ");
		System.out.println();*/
		return divS;//*Math.log(doc.size());
	}
	
	public double compuDiver(ArrayList<String> doc)
	{
		double divS = 0.0;
		
		//ArrayList<String> rdoc = removeHardWords(doc);
		
		ArrayList<String> diff = new ArrayList<String>();
		
		ArrayList<Integer> freArr = normBagOfWords(doc, diff);
		
		//int docLen = doc.size();
		//double sisThreshold = ;
		
		ArrayList<Double> sis = new ArrayList<Double>();
		ArrayList<Integer> indArr = new ArrayList<Integer>();
		
		computDiffSis(diff,sis,indArr);
		
		sortSis(sis,indArr);
		
		ArrayList<ArrayList<String>> clustArr = computCluster(diff,sis,indArr);
		
		
		//System.out.println("the score of diversity: " + divS);
		/*for(int i=0; i<clustArr.size();i++)
			if(clustArr.get(i).size()<2)
			{
				clustArr.remove(i);
				i--;
			}*/
		//for(int i=0; i<clustArr.size();i++)
			//System.out.println(clustArr.get(i).toString());
		
		divS = diverMeasure(clustArr,diff,freArr);
		
		return divS;//*Math.log(doc.size());
	}
	
	
	public ArrayList<String> readFile(String path)
	{
		try{
			
			ArrayList<String> partStr = new ArrayList<String>();
			
//			List<List<HasWord>> sentences = MaxentTagger.tokenizeText(new BufferedReader(new FileReader(path)));
//			
//			//int textLen = 0;
//			
//			//for (List<HasWord> sentence : sentences)
//				//textLen += sentence.size(); 
//			ArrayList<Integer> textList = new ArrayList<Integer>();
//			
//			for (List<HasWord> sentence : sentences)
//			{
//				 ArrayList<TaggedWord> tSentence = tagger.tagSentence(sentence);
//			     
//				 //System.out.println(tSentence.toString());
//			       
//			      for(int j=0; j<tSentence.size(); j++)
//			      {
//			    	 // String tag = tSentence.get(j).tag();
//			    	  
//			    	  // System.out.println(tag+ " "+ tSentence.get(j).value());
//			    	  //if(tag.length()>=2 && tag.substring(0, 2).equals("NN"))
//			    	  //{
//			    		  String word = tSentence.get(j).value();
//			  			
//				    	  //	String token = m_Stemmer.stemString(word);
//				    	  	String token = word.toLowerCase();
//			    		 // String token = word;
//				    	  	Matcher m = p.matcher(token); // only save these strings only contains characters
//				    	  	if( !m.find()  && token.length()>2 && token.length()<25 )
//				    	  	//if( Character.isLetter(token.charAt(0))  && token.length()>2 )
//					    	//{
//				    			  if (!m_EnStopwords.isStopword(token)) 
//				    			  {
//				    				  partStr.add(token);
//				    			  }
//					    	//}
//			    	  	//System.out.println();
//			    	  	//String word = tSentence.get(j).value();
//			    	  //}
//			      }
//			}
			return partStr;
			
			
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		System.out.println("Warning!");
		return null;
	}
	
	
	public void test()
	{
		String s1 = "digital";
		String s2 = "media";
		String s3 = "display";
		String s4 = "image";
		//String s5 = "quarter";
		String s6 = "operating";
		String s7 = "system";
		String s8 = "device";
		//String s9 = "perform";
		ArrayList<String> arr = new ArrayList<String>();
		
		arr.add(s1);
		arr.add(s2);
		arr.add(s3);
		arr.add(s4);
		//arr.add(s5);
		arr.add(s6);
		arr.add(s7);
		arr.add(s8);
		//arr.add(s9);
		
		//compuDiver(arr);
		
	}
	
	public void testFile(String path) throws Exception
	{
		ArrayList<String> textArr = readFile(path);
		
		//compuDiver(textArr);
	}
	
	public double testPse(String path1, String path2)throws Exception
	{
		ArrayList<String> doc1 = readFile(path1);
		ArrayList<String> doc2 = readFile(path2);
		ArrayList<String> doc = new ArrayList<String>();
		
		doc.addAll(doc1.subList(0, 100));
		doc.addAll(doc2.subList(0, 100));
		//System.out.println(doc.size());
		double div = compuDiverBasedKmeans(doc);
		//System.out.println(div);
		return div;
	}
	
	public void choosePara() throws Exception
	{
		readVector();
		
		for(double bigP = 0.4; bigP<=0.75; bigP += 0.05)
		{
			for(double smallP=0.15; smallP<=0.4; smallP += 0.05)
			{
				initialPara(bigP,smallP);
				System.out.println("bigP: "+ bigP + " smallP: "+ smallP);
				
				testPse("C:/Users/jipeng/Desktop/dataset/BBC/bbc/tech/001.txt","C:/Users/jipeng/Desktop/dataset/BBC/bbc/tech/002.txt");
				testPse("C:/Users/jipeng/Desktop/dataset/BBC/bbc/tech/001.txt","C:/Users/jipeng/Desktop/dataset/BBC/bbc/sport/001.txt");
				testPse("C:/Users/jipeng/Desktop/dataset/BBC/bbc/tech/001.txt","C:/Users/jipeng/Desktop/dataset/BBC/bbc/entertainment/001.txt"); 
			}
		}
	}
	
	public void oneTime() throws Exception {
		
		
		readVector();
		//initialPara(0.55,0.25);
		//ed.test();
		/*ed.testFile("C:/Users/jipeng/Desktop/dataset/BBC/bbc/business/002.txt");
		System.out.println("------------");
		ed.testFile("C:/Users/jipeng/Desktop/dataset/BBC/bbc/entertainment/004.txt");
		System.out.println("------------");
		ed.testFile("C:/Users/jipeng/Desktop/dataset/BBC/bbc/sport/004.txt");
		System.out.println("------------");
		ed.testFile("C:/Users/jipeng/Desktop/dataset/BBC/bbc/tech/004.txt");*/
		System.out.println("********************");
		testPse("C:/Users/jipeng/Desktop/dataset/BBC/bbc/tech/001.txt","C:/Users/jipeng/Desktop/dataset/BBC/bbc/tech/002.txt");
		System.out.println("********************");
		testPse("C:/Users/jipeng/Desktop/dataset/BBC/bbc/tech/001.txt","C:/Users/jipeng/Desktop/dataset/BBC/bbc/sport/009.txt");
		//System.out.println("********************");
		//ed.testPse("C:/Users/jipeng/Desktop/dataset/BBC/bbc/tech/002.txt","C:/Users/jipeng/Desktop/dataset/BBC/bbc/entertainment/002.txt");
		System.out.println("********************");
		testPse("C:/Users/jipeng/Desktop/dataset/BBC/bbc/tech/001.txt","C:/Users/jipeng/Desktop/dataset/BBC/bbc/entertainment/007.txt"); 
		//ed.readFile("C:/Users/jipeng/Desktop/dataset/BBC/bbc/business/001.txt");
		//ed.wordDist("", w2)
		//ed.testWordDistance("C:/Users/jipeng/Desktop/dataset/BBC/005.txt");
		//ed.testVector();
		//ed.compuDiverBasedClustering("C:/Users/jipeng/Desktop/dataset/001.txt", "C:/Users/jipeng/Desktop/dataset/002.txt");
		//ed.compuDiverBasedClustering("C:/Users/jipeng/Desktop/dataset/BBC/bbc/business/002.txt", "C:/Users/jipeng/Desktop/dataset/BBC/bbc/business/003.txt");
		//ed.compuDiverBasedClustering("C:/Users/jipeng/Desktop/dataset/BBC/bbc/business/002.txt", "C:/Users/jipeng/Desktop/dataset/BBC/bbc/entertainment/001.txt");
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		try
		{
			WordClusteringForDiver ed = new WordClusteringForDiver();
			//ed.choosePara();
			ed.oneTime();
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		
	}

}
