package kex.stopwords;



import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Hashtable;

/**
 * Class that can test whether a given string is a stop word.
 * Lowercases all words before the test.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version 1.0
 */
public class StopwordsChinese extends Stopwords {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/** The hashtable containing the list of stopwords */
	private static Hashtable m_Stopwords = null;
	
	
	static {
		
		if (m_Stopwords == null) {
			m_Stopwords = new Hashtable();
			Double dummy = new Double(0);
			
			/** 英文停用词 */
			File txt = new File("data/stopwords/stopwords_en.txt");	
			InputStreamReader is;
			String sw = null;
			try {
				is = new InputStreamReader(new FileInputStream(txt));
				BufferedReader br = new BufferedReader(is);				
				while ((sw=br.readLine()) != null)  {					
					m_Stopwords.put(sw, dummy);   
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			/** 中文停用词 */
			txt = new File("data/stopwords/stopwords_ch.txt");	
			try {
				is = new InputStreamReader(new FileInputStream(txt));
				BufferedReader br = new BufferedReader(is);				
				while ((sw=br.readLine()) != null)  {					
					m_Stopwords.put(sw, dummy);   
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			
		}
	}
	
	/** 
	 * Returns true if the given string is a stop word.
	 */
	public boolean isStopword(String str) {
		
		return m_Stopwords.containsKey(str.toLowerCase());
	}
}
		
		
