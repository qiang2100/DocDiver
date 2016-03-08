package ldaDiver;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Gibbs sampler for estimating the best assignments of topics for words and
 * documents in a corpus. The algorithm is introduced in Tom Griffiths' paper
 * "Gibbs sampling in the generative model of Latent Dirichlet Allocation"
 * (2002).
 * 
 * @author heinrich
 */
public class LDADiver {

    /**
     * document data (term lists)
     */
    //int[][] documents;

    /**
     * vocabulary size
     */
    int V;

    /**
     * number of topics
     */
    int K;

    /**
     * Dirichlet parameter (document--topic associations)
     */
    double alpha=0.1;

    /**
     * Dirichlet parameter (topic--term associations)
     */
    double beta=0.1;

    /**
     * topic assignments for each word.
     */
    int z[][];


    /**
     * cwt[i][j] number of instances of word i (term?) assigned to topic j.
     */
    int[][] nw;


    /**
     * na[i][j] number of words in document i assigned to topic j.
     */
    int[][] nd;


    /**
     * nwsum[j] total number of words assigned to topic j.
     */
    int[] nwsum;
   
    /**
     * nasum[i] total number of words in document i.
     */
    int[] ndsum;
  
    /**
     * cumulative statistics of theta
     */
    double[][] thetasum;

    
    /**
     * cumulative statistics of phi
     */
    double[][] phisum;
 
    /**
     * size of statistics
     */
    int numstats;

    
    /**
     * sampling lag (?)
     */
    private static int THIN_INTERVAL = 20;

    /**
     * burn-in period
     */
    private static int BURN_IN = 100;

    /**
     * max iterations
     */
    private static int ITERATIONS = 1000;

    /**
     * sample lag (if -1 only one sample taken)
     */
    private static int SAMPLE_LAG;

    private static int dispcol = 0;

   // ArrayList<Integer> label ;
	//ArrayList<String> gene ;
	ArrayList<ArrayList<Integer>> sData ;
	
	ArrayList<String> wordsArr = new ArrayList<String>();
	ArrayList<Integer> lablesArr = new ArrayList<Integer>();
	
	int clu[];
	//HashMap<Integer,Integer> gene2Id = new HashMap<Integer,Integer>();
    /**
     * Initialise the Gibbs sampler with data.
     * 
     * @param V
     *            vocabulary size
     * @param data
     */
    public LDADiver() {

       // this.documents = documents;
       // this.V = V;
    }

    /**
     * Initialisation: Must start with an assignment of observations to topics ?
     * Many alternatives are possible, I chose to perform random assignments
     * with equal probabilities
     * 
     * @param K
     *            number of topics
     * @return z assignment of topics to words
     */
    public void initialState(int K) {
     //   int i;

        int M = sData.size();

        // initialise count variables.
        nw = new int[V][K];
        nd = new int[M][K];
        nwsum = new int[K];
       ndsum = new int[M];

        // The z_i are are initialised to values in [1,K] to determine the
        // initial state of the Markov chain.

        z = new int[M][];
        
        clu = new int[M];
        for (int m = 0; m < M; m++) {
            int N = sData.get(m).size();
            z[m] = new int[N];
            for (int n = 0; n < N; n++) {
                int topic = (int) (Math.random() * K);
                z[m][n] = topic;
                // number of instances of word i assigned to topic j
                nw[sData.get(m).get(n)][topic]++;
                // number of words in document i assigned to topic j.
                nd[m][topic]++;
                // total number of words assigned to topic j.
                nwsum[topic]++;
            }
            // total number of words in document i
            ndsum[m] = N;
        }
        
      
    }

   
    
    /**
     * Main method: Select initial state ? Repeat a large number of times: 1.
     * Select an element 2. Update conditional on other elements. If
     * appropriate, output summary for each run.
     * 
     * @param K
     *            number of topics
     * @param alpha
     *            symmetric prior parameter on document--topic associations
     * @param beta
     *            symmetric prior parameter on topic--term associations
     */
    private void gibbs(int K, double alpha, double beta) {
        this.K = K;
        this.alpha = alpha;
        this.beta = beta;

        // init sampler statistics
        if (SAMPLE_LAG > 0) {
            thetasum = new double[sData.size()][K];
            phisum = new double[K][V];
            numstats = 0;
        }

        // initial state of the Markov chain:
        initialState(K);

        System.out.println("Sampling " + ITERATIONS
            + " iterations with burn-in of " + BURN_IN + " (B/S="
            + THIN_INTERVAL + ").");

        for (int i = 0; i < ITERATIONS; i++) {

            // for all z_i
            for (int m = 0; m < z.length; m++) {
                for (int n = 0; n < z[m].length; n++) {

                    // (z_i = z[m][n])
                    // sample from p(z_i|z_-i, w)
                    int topic = sampleFullConditional(m, n);
                    z[m][n] = topic;
                }
            }

            if ((i < BURN_IN) && (i % THIN_INTERVAL == 0)) {
                System.out.print("B");
                dispcol++;
            }
            // display progress
            if ((i > BURN_IN) && (i % THIN_INTERVAL == 0)) {
                System.out.print("S");
                dispcol++;
            }
            // get statistics after burn-in
            if ((i > BURN_IN) && (SAMPLE_LAG > 0) && (i % SAMPLE_LAG == 0)) {
                updateParams();
                System.out.print("|");
                if (i % THIN_INTERVAL != 0)
                    dispcol++;
            }
            if (dispcol >= 100) {
                System.out.println();
                dispcol = 0;
            }
        }
    }

   
  
    /**
     * Sample a topic z_i from the full conditional distribution: p(z_i = j |
     * z_-i, w) = (n_-i,j(w_i) + beta)/(n_-i,j(.) + W * beta) * (n_-i,j(d_i) +
     * alpha)/(n_-i,.(d_i) + K * alpha)
     * 
     * @param m
     *            document
     * @param n
     *            word
     */
    private int sampleFullConditional(int m, int n) {

        // remove z_i from the count variables
        int topic = z[m][n];
        nw[sData.get(m).get(n)][topic]--;
        nd[m][topic]--;
        nwsum[topic]--;
        //ndsum[m]--;

        // do multinomial sampling via cumulative method:
        double[] p = new double[K];
        for (int k = 0; k < K; k++) {
            p[k] = (nw[sData.get(m).get(n)][k] + beta) / (nwsum[k] + V * beta)
                * (nd[m][k] + alpha);// / (ndsum[m] + K * alpha);
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
        nw[sData.get(m).get(n)][topic]++;
        nd[m][topic]++;
        nwsum[topic]++;
       // ndsum[m]++;

        return topic;
    }

    
    
    /**
     * Add to the statistics the values of theta and phi for the current state.
     */
    private void updateParams() {
        for (int m = 0; m < sData.size(); m++) {
            for (int k = 0; k < K; k++) {
                thetasum[m][k] += (nd[m][k] + alpha) / (ndsum[m] + K * alpha);
            }
        }
        for (int k = 0; k < K; k++) {
            for (int w = 0; w < V; w++) {
                phisum[k][w] += (nw[w][k] + beta) / (nwsum[k] + V * beta);
            }
        }
        numstats++;
    }

  
    /**
     * Retrieve estimated document--topic associations. If sample lag > 0 then
     * the mean value of all sampled statistics for theta[][] is taken.
     * 
     * @return theta multinomial mixture of document topics (M x K)
     */
    public double[][] getTheta() {
        double[][] theta = new double[sData.size()][K];

        if (SAMPLE_LAG > 0) {
            for (int m = 0; m < sData.size(); m++) {
                for (int k = 0; k < K; k++) {
                    theta[m][k] = thetasum[m][k] / numstats;
                }
            }

        } else {
            for (int m = 0; m < sData.size(); m++) {
                for (int k = 0; k < K; k++) {
                    theta[m][k] = (nd[m][k] + alpha) / (ndsum[m] + K * alpha);
                }
            }
        }

        return theta;
    }

   
    /**
     * Retrieve estimated topic--word associations. If sample lag > 0 then the
     * mean value of all sampled statistics for phi[][] is taken.
     * 
     * @return phi multinomial mixture of topic words (K x V)
     */
    public double[][] getPhi() {
        double[][] phi = new double[K][V];
        if (SAMPLE_LAG > 0) {
            for (int k = 0; k < K; k++) {
                for (int w = 0; w < V; w++) {
                    phi[k][w] = phisum[k][w] / numstats;
                }
            }
        } else {
            for (int k = 0; k < K; k++) {
                for (int w = 0; w < V; w++) {
                    phi[k][w] = (nw[w][k] + beta) / (nwsum[k] + V * beta);
                }
            }
        }
        return phi;
    }

   
    /**
     * Configure the gibbs sampler
     * 
     * @param iterations
     *            number of total iterations
     * @param burnIn
     *            number of burn-in iterations
     * @param thinInterval
     *            update statistics interval
     * @param sampleLag
     *            sample interval (-1 for just one sample at the end)
     */
    public void configure(int iterations, int burnIn, int thinInterval,
        int sampleLag) {
        ITERATIONS = iterations;
        BURN_IN = burnIn;
        THIN_INTERVAL = thinInterval;
        SAMPLE_LAG = sampleLag;
    }

    public void readCSV( String path)
    {
    	String csvFile = path;
    	BufferedReader br = null;
		String line = "";
		String cvsSplitBy = " ";
		
		//label = new ArrayList<Integer>();
		
		//gene = new ArrayList<String>();
		
		sData = new ArrayList<ArrayList<Integer>>();
		
	
		try {
	 
			br = new BufferedReader(new FileReader(csvFile));
			
			
			//V = ge.length-1;
			
			HashMap<Integer,Integer> vArr = new HashMap<Integer,Integer>();
			int id=0;
			
			while ((line = br.readLine()) != null) {
	 
			        // use comma as separator
				String[] num = line.split(cvsSplitBy);
				
				//System.out.println(num[0]);
				//int laberNum = Integer.parseInt(num[0]);
				
				//label.add(laberNum);
					
				ArrayList<Integer> sample = new ArrayList<Integer>();
		
				
				for(int i=0; i<num.length; i++)
				{
					if(num[i].equals(""))
						continue;
					int word = Integer.parseInt(num[i]);
					
					if(!vArr.containsKey(word))
					{
						vArr.put(word, id);
						sample.add(id);
						id++;
						
					}else
					{
						sample.add(vArr.get(word));
					}
					
				
				}
				sData.add(sample);
				
			}
			V = vArr.size();
	 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	 
		
    }
    
   
    
    public void runLDA(int v, ArrayList<ArrayList<Integer>> currArr, int topic)
    {
    	int K = topic;
    	
    	V = v;
    	
    	sData = currArr;
    	
    	 double alpha = .1;
         double beta = .1;
    	
         configure(1000, 100, 100, 5);
         
         gibbs(K, alpha, beta);
         
         
    }
    
  
    
    public void runLDATest(ArrayList<ArrayList<Integer>> currArr)
    {
    	
    	sData = currArr;
    	
    	// double alpha = .1;
   //      double beta = .1;
    	
        // configure(1000, 100, 100, 5);  
    }
    
    
    
    public void readWord( String path)
    {
    	
    	BufferedReader br = null;
		String line = "";
		String cvsSplitBy = " ";
		
		try {
	 
			br = new BufferedReader(new FileReader(path));

			//ArrayList<Integer> vArr = new ArrayList<Integer>();
			
			while ((line = br.readLine()) != null) {
	 
			        // use comma as separator
				String[] num = line.split(cvsSplitBy);
				
				//System.out.println(num[0]);
				//int laberNum = Integer.parseInt(num[0]);
				
				//label.add(laberNum);
				//wordId.put(Integer.parseInt(num[0]), num[1]);
				wordsArr.add(num[0]);
			}
			V = wordsArr.size();
	 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	 
		
    }
    
    public void readLable( String path)
    {
    	
    	BufferedReader br = null;
		String line = "";
		String cvsSplitBy = " ";
		
		try {
	 
			br = new BufferedReader(new FileReader(path));

			//ArrayList<Integer> vArr = new ArrayList<Integer>();
			
			while ((line = br.readLine()) != null) {
	 
			        // use comma as separator
				String[] num = line.split(cvsSplitBy);
				
				//System.out.println(num[0]);
				//int laberNum = Integer.parseInt(num[0]);
				
				//label.add(laberNum);
				//wordId.put(Integer.parseInt(num[0]), num[1]);
				lablesArr.add(Integer.parseInt(num[0]));
			}
			//V = wordsArr.size();
	 
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
    }
    
    public void callLDA(String path, int topics) throws Exception
    {
    	
    	readCSV(path);
    	
    	//readLable(lablePath);
    	
    	
    //	int M = sData.size();
    	
    	 int K = topics;
    	// V = gene2Id.size();
    	 //System.out.println("v : " + V);
         // good values alpha = 2, beta = .5
         double alpha = .1;
         double beta = .1;

         System.out.println("Latent Dirichlet Allocation using Gibbs Sampling.");

         // LdaGibbsSampler lda = new LdaGibbsSampler(documents, V);
          configure(4000, 2000, 100, 5);
          
      //    ArrayList<Double> nmiArr = new ArrayList<Double>();
          
          //String resPath = "C:/Users/qjp/Desktop/TopicModel/dataset/googlenews/result/topics=" + topics + "NMI.txt";
          
         // FileWriter fw = new FileWriter(resPath);
  		  //BufferedWriter bw = new BufferedWriter(fw);
  		  
          
          gibbs(K, alpha, beta);

         
         // double[][] theta = getTheta();
          //  double[][] phi = getPhi();
           
       
         //bw.close();
        // fw.close();
        
        // test();
    }
    
    public double getDistanceBasedCosine(int topic1, int topic2)
    {
    	double sis = 0;
    	double sisT1 = 0;
    	double sisT2 = 0;
    	
    	
    	for(int i=0; i<nd.length; i++)
    	{
    		sis += nd[i][topic1]*nd[i][topic2];
    		
    		sisT1 += nd[i][topic1]*nd[i][topic1];
    		
    		sisT2 += nd[i][topic2]*nd[i][topic2];
    	}
    	
    	sis = sis/(Math.sqrt(sisT1)*Math.sqrt(sisT2));
    	
    	return 1/sis;	
    }
    
   /* public double getDistanceBasedProb(int topic1, int topic2)
    {
    	double sis = 0;
    	double sisT1 = 0;
    	double sisT2 = 0;
    	
    	for(int i=0; i<nd.length; i++)
    	{
    		sis += nd[i][topic1]*nd[i][topic2];
    		
    		sisT1 += nd[i][topic1]*nd[i][topic1];
    		
    		sisT2 += nd[i][topic2]*nd[i][topic2];
    	}
    	sis = sis/(Math.sqrt(sisT1)*Math.sqrt(sisT2));
    	return 1.0/sis;	
    }
    */
    
    /* Input: the index of one document; Output: the diversity score of this document */
    public double computDivBasedCosine(int index1, int index2)
    {
    	int comNd[] = new int[K];
    	
    	int totalWord = 0;
    	
    	for(int i=0; i<K; i++)
    	{
    		//comNd[i] = Math.round((nd[index1][i]+nd[index2][i])/2);
    		comNd[i] = (nd[index1][i]+nd[index2][i])/2;
    		totalWord += nd[index1][i] + nd[index2][i];
    	}
    	
    	double div = 0;
    	
    	for(int i=0; i<K; i++)
    	{
    		for(int j=0; j<K; j++)
    		{
    			div += ((double)comNd[i]/totalWord)*((double)comNd[j]/totalWord)*getDistanceBasedCosine(i,j);
    			/*if(i==j)
    				System.out.println("i==j" + getDistanceBasedCosine(i,j));
    			else
    				System.out.println("i!=j" + getDistanceBasedCosine(i,j));*/
    		}
    	}
    	
    	return div;
    }
    
   /* public double computDivBasedProb(int ind)
    {
    	double div = 0;
    	
    	for(int i=0; i<K; i++)
    	{
    		for(int j=0; j<K; j++)
    		{
    			
    		}
    	}
    	
    	return div;
    	
    	
    }*/
    
 
    
    
    public void mainFun() throws Exception
    {
    	  String path = "C:/Users/qjp/Desktop/TopicModel/dataset/NIPS/texts10.txt";
    	  int k = 10;
			//int topWords =20; 
		
    	  callLDA(path,k);
    	  
    	  double div1 = computDivBasedCosine(1,2);
    	  System.out.println(div1);
    	  
    	  double div2 = computDivBasedCosine(1,1001);
    	  System.out.println(div2);
    }
    
    public void mainFunFor() throws Exception
    {
    	  String path = "C:/Users/qjp/Desktop/TopicModel/dataset/NIPS/texts10.txt";
    	  int k = 10;
			//int topWords =20; 
		
    	  callLDA(path,k);
    	  
    	  double div1 = computDivBasedCosine(1,2);
    	  System.out.println(div1);
    	  
    	  double div2 = computDivBasedCosine(1,1001);
    	  System.out.println(div2);
    }
    
    /**
     * Driver with example data.
     * 
     * @param args
     */
    public static void main(String[] args) {

    		LDADiver lda = new LDADiver();
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
    			  
    			 // lda.mainFun();
    			  
    			 // lda.percentIterMain();
    	          //lda.mainFun(path, lablePath, k);
    			// lda.iterMain(path, wordPath);
    		}catch(Exception e)
    		{
    			e.printStackTrace();
    		}
        
    	
    }

 
}
