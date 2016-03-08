package textcluster;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public  class Program
    {
       public static void main(String[] args) throws IOException
        {
            //1����ȡ�ĵ�����
            String[] docs = getInputDocs("d:/input.txt");
            
            if (docs.length < 1)
            {
                System.out.println("û���ĵ�����");
                System.in.read();
              //  System.exit(0);
                return;
            }
            /*else{
            	for(String s:docs){
            		System.out.println(s);
            	}
            }*/

            //2����ʼ��TFIDF���������������ÿ���ĵ���TFIDFȨ��
            TFIDFMeasure tf = new TFIDFMeasure(docs, new Tokeniser());
            System.out.println(tf.get_numTerms());
          

            int K = 2; //�۳�3������

            //3�����k-means��������ݣ���һ���������飬��һά��ʾ�ĵ�����
            //�ڶ�ά��ʾ�����ĵ��ֳ��������д�
            float[][] data = new float[docs.length][];
            int docCount = docs.length; //�ĵ�����
            int dimension = tf.get_numTerms();//���дʵ���Ŀ
            for (int i = 0; i < docCount; i++)
            {
                for (int j = 0; j < dimension; j++)
                {
                    data[i] = tf.GetTermVector2(i); //��ȡ��i���ĵ���TFIDFȨ������
                }
            }

            //4����ʼ��k-means�㷨����һ�������ʾ������ݣ��ڶ��������ʾҪ�۳ɼ�����
            WawaKMeans kmeans = new WawaKMeans(data, K);
            //5����ʼ���
            kmeans.Start();

            //6����ȡ���������
            WawaCluster[] clusters = kmeans.getClusters();
            for(WawaCluster cluster : clusters){

                List<Integer> members = cluster.CurrentMembership;
                System.out.println("-----------------");
                for (int i =0; i<members.size(); i++)
                {
                	System.out.println(members.get(i));
                }

            
            }
            /*foreach (WawaCluster cluster in clusters)
            {
                List<int> members = cluster.CurrentMembership;
                Console.WriteLine("-----------------");
                foreach (int i in members)
                {
                    Console.WriteLine(docs[i]);
                }

            }*/
           // System.in.read();
           // Console.Read();
        }

        /// <summary>
        /// ��ȡ�ĵ�����
        /// </summary>
        /// <returns></returns>
        private static String[] getInputDocs(String file)
        {
            List<String> ret = new ArrayList<String>();
            
            try
            {
                BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(file)));
                {
                    String temp;
                    while ((temp = br.readLine()) != null)
                    {
                        ret.add(temp);
                    }
                }
            }
            catch (Exception ex)
            {
                ex.printStackTrace();
            }
            String[] fileString=new String[ret.size()];
            return (String[]) ret.toArray(fileString);
        }
    }
