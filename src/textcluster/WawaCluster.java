package textcluster;

import java.util.ArrayList;
import java.util.List;

     public class WawaCluster
    {
        public WawaCluster(int dataindex,float[] data)
        {
            CurrentMembership.add(dataindex);
            Mean = data;
        }

        /// <summary>
        /// �þ������ݳ�Ա����
        /// </summary>
        public List<Integer> CurrentMembership = new ArrayList<Integer>();
        /// <summary>
        /// �þ��������
        /// </summary>
         float[] Mean;
        /// <summary>
        /// �÷�������������ľ�ֵ 
        /// </summary>
        /// <param name="coordinates"></param>
        public void UpdateMean(float[][] coordinates)
        {
            // ��� mCurrentMembership ȡ��ԭʼ���ϵ���� coord ���ö����� coordinates ��һ���Ӽ���
            //Ȼ��ȡ�����Ӽ��ľ�ֵ��ȡ��ֵ���㷨�ܼ򵥣����԰� coordinates �����һ�� m*n �ľ��� ,
            //ÿ����ֵ����ÿ�������е�ȡ��ƽ��ֵ , //��ֵ������ mCenter ��

            for (int i = 0; i < CurrentMembership.size(); i++)
            {
                float[] coord = coordinates[CurrentMembership.get(i)];
                for (int j = 0; j < coord.length; j++)
                {
                    Mean[j] += coord[j]; // �õ�ÿ�������еĺͣ�
                }
                for (int k = 0; k < Mean.length; k++)
                {
                    Mean[k] /= coord.length; // ��ÿ��������ȡƽ��ֵ
                }
            }
        }
    }

