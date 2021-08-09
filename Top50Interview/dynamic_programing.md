[Back](/discuss/interview-question?currentPage=1&orderBy=hot&query=)

##### Template For Dynamic programming

**Dynmaic Programming For Practice**

Sharing some topic wise good Dynamic Programming problems and sample solutions to observe on how to approach.

1.**Unbounded Knapsack** **or Target sum**  
Identify if problems talks about finding groups or subset which is equal to given target.

[https://leetcode.com/problems/target-sum/](https://leetcode.com/problems/target-sum/)  
[https://leetcode.com/problems/partition-equal-subset-sum/](https://leetcode.com/problems/partition-equal-subset-sum/)  
[https://leetcode.com/problems/last-stone-weight-ii/](https://leetcode.com/problems/last-stone-weight-ii/)  
[https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/)

All the above problems can be solved by 01 Knapsack or Target sum algo with minor tweaks.  
Below is a standard code for 01 knapsack or target sum problems.

```java
int 01 knacsack(vector<int>& nums,vector<int>& v, int w)  // nums array , w total amount that have to collect 
	{                                                     // v value array
		int n=nums.size();
		
		vector<vector<bool>> d(n+1,vector<bool>(w+1,0));  
		for(int i=1;i<=n;i++)
		{
			for(int j=1;j<=w;j++)
			{
				if(j<nums[i-1])  
				{
					d[i][j]=d[i-1][j];
				}
				else if(nums[i-1]<=j)
				{
					d[i][j]=max(v[i-1]+d[i-1][j-nums[i-1]],d[i-1][j]);
				}
			}
		}
		
		return d[n][w];
		
	}
    
``` 

**Funtion for Target sum**
```java

    int countsubset(vector<int>& nums, int w) 
        {
    	    int n=nums.size();
    		
            vector<vector<bool>> d(n+1,vector<bool>(w+1));
            for(int i=0;i<=n;i++)
            {
                    d[i][0]=1;
            }
            for(int i=1;i<=w;i++)
            {
                    d[0][i]=0;
            }
            
            for(int i=1;i<=n;i++)
            {
                for(int j=1;j<=w;j++)
                {
                    if(j<nums[i-1])
                    {
                        d[i][j]=d[i-1][j];
                    }
                    else if(nums[i-1]<=j)
                    {
                        d[i][j]=d[i-1][j-nums[i-1]] + d[i-1][j];
                    }
                }
            }
            
            return d[n][w];
        }
    
```

2.**Unbounded Knapsack**  
Identify if problems talks about finding groups or subset which is equal to given target and repetition is allowed.

[https://leetcode.com/problems/coin-change-2/](https://leetcode.com/problems/coin-change-2/)  
[https://leetcode.com/problems/coin-change/](https://leetcode.com/problems/coin-change/)

All the above problems can be solved by unbounded Knapsack algo with minor tweaks.  
Below is a standard code for 01 knapsack or target sum problems.

```java
    int unboundedknacsack(vector<int>& nums,vector<int>& v, int w) 
        {
            int n=nums.size();
    		
            vector<vector<bool>> d(n+1,vector<bool>(w+1,0));
            for(int i=1;i<=n;i++)
            {
                for(int j=1;j<=w;j++)
                {
                    if(j<nums[i-1])
                    {
                        d[i][j]=d[i-1][j];
                    }
                    else if(nums[i-1]<=j)
                    {
                        d[i][j]=max(v[i-1]+d[i][j-v[i-1]],d[i-1][j]);
                    }
                }
            }
            
            return d[n][w];
        }
    
```

**or**

```java
           int change(int amount, vector<int>& coins) 
        {
           vector<vector<int>> d(coins.size()+1,vector<int>(amount+1));
            
           for(int i=0;i<=coins.size();i++)
           {
               d[i][0]=1;
           }
            for(int i=1;i<=amount;i++)
           {
               d[0][i]=0;
           }
            
            for(int i=1;i<=coins.size();i++)
           {
               for(int j=1;j<=amount;j++)
               {
                   if(j<coins[i-1])
                   {
                       d[i][j]=d[i-1][j];
                   }
                   
                   else if(j>=coins[i-1])
                   {
                       d[i][j]=(d[i][j-coins[i-1]]+d[i-1][j]);
                   }
               }
           }
            
            return d[coins.size()][amount]; 
        }
    
```

3.**Longest Increasing Subsequence (LIS)**

Identify if problems talks about finding longest increasing subset.

[https://leetcode.com/problems/minimum-cost-to-cut-a-stick/](https://leetcode.com/problems/minimum-cost-to-cut-a-stick/)  
[https://leetcode.com/problems/longest-increasing-subsequence/](https://leetcode.com/problems/longest-increasing-subsequence/)  
[https://leetcode.com/problems/largest-divisible-subset/](https://leetcode.com/problems/largest-divisible-subset/)  
[https://leetcode.com/problems/perfect-squares/](https://leetcode.com/problems/perfect-squares/)  
[https://leetcode.com/problems/super-ugly-number/](https://leetcode.com/problems/super-ugly-number/)

[https://leetcode.com/problems/russian-doll-envelopes/](https://leetcode.com/problems/russian-doll-envelopes/)  
[https://leetcode.com/problems/maximum-height-by-stacking-cuboids/description/](https://leetcode.com/problems/maximum-height-by-stacking-cuboids/description/)

@Nam\_22 mentioning above two question .

All the above problems can be solved by longest Increasing subsequence algo with minor tweaks.  
Below is a standard code for LIS problems.

    
    
```java
    
        int lengthOfLIS(vector<int>& nums) 
        {
            vector<int> d(nums.size(),1);
            
            int m=0;
            for(int i=0;i<nums.size();i++)
            {
                for(int j=0;j<i;j++)
                {
                    if(nums[j]<nums[i] && d[i]<d[j]+1)
                    {
                        d[i]=d[j]+1;
                    }
                }
                m=max(d[i],m);
            }
        
            return m;
        }
    
```
    

**longest bitonic subsequence**

```java
    int lbs(vector<int> v)
    {
        vector<int> lis(v.size(),1);
        vector<int> lds(v.size(),1);
        
       for(int i=0;i<v.size();i++)
        {
            
            for(int j=0;j<i;j++)
            {
                if(v[j]<v[i] && lis[i]<lis[j]+1)
                {
                 lis[i]=lis[j]+1;
                }
            }
    
        }
        
       for(int i=v.size()-2;i>0;i--)
        {
           
           for(int j=v.size()-1;j>i;j--)
           {
               if(v[j]<v[i] && lds[i]<lds[j]+1)
              { 
                 lds[i]=lds[j]+1;
              }
           }
           
        }
        
        int m=0;
        for(int i=0;i<v.size();i++)
        { 
            m=max(m,lis[i]+lds[i]-1);
        }
        
        return m;
    }
    
```

4.**Longest Common Subsequence**

Identify if problems talks about finding longest common subset.

1.**subsequence**  
[https://leetcode.com/problems/longest-common-subsequence/](https://leetcode.com/problems/longest-common-subsequence/)  
[https://leetcode.com/problems/distinct-subsequences/](https://leetcode.com/problems/distinct-subsequences/)  
[https://leetcode.com/problems/shortest-common-supersequence/](https://leetcode.com/problems/shortest-common-supersequence/)  
[https://leetcode.com/problems/distinct-subsequences/](https://leetcode.com/problems/distinct-subsequences/)  
[https://leetcode.com/problems/interleaving-string/](https://leetcode.com/problems/interleaving-string/)

```java
     int longestCommonSubsequence(string text1, string text2) 
        {
            int n1 = text1.size();
            int n2 = text2.size();
            vector<vector<int>> dp(n1+1,vector<int>(n2+1,0));
            
    
            for(int i=1;i<=n1;i++)
            {
                for(int j=1;j<=n2;j++)
                {
                    if(text1[i-1] == text2[j-1])
                        dp[i][j] = 1+dp[i-1][j-1];
                    else
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
    
                }
            }
            return dp[n1][n2];
    
        }
    
```

2.**substring**

[https://leetcode.com/problems/maximum-length-of-repeated-subarray/](https://leetcode.com/problems/maximum-length-of-repeated-subarray/)

```java
     int longestCommonSubstring(string text1, string text2) 
        {
            int n1 = text1.size();
            int n2 = text2.size();
            vector<vector<int>> dp(n1+1,vector<int>(n2+1,0));
            
            int r=0;
            for(int i=1;i<=n1;i++)
            {
                for(int j=1;j<=n2;j++)
                {
                    if(text1[i-1] == text2[j-1])
                       { 
                         dp[i][j] = 1+dp[i-1][j-1];
                         r=max(dp[i][j],r);
                       }
                    else
                        dp[i][j] = 0;
    
                }
            }
            return dp[n1][n2];
    
        }
    
```

3.**palindrome**

[https://leetcode.com/problems/longest-palindromic-substring/](https://leetcode.com/problems/longest-palindromic-substring/)  
[https://leetcode.com/problems/longest-palindromic-subsequence/](https://leetcode.com/problems/longest-palindromic-subsequence/)  
[https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/](https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/)  
[https://leetcode.com/problems/delete-operation-for-two-strings/](https://leetcode.com/problems/delete-operation-for-two-strings/)

```java
    int lps(string s1)
    {
        int n1=s1.length();
        string s2=s1;
        reverse(s2.begin(),s2.end());
        int n2=s2.length();
        
        vector<vector<int>> dp(n1+1,vector<int>(n2+1,0));
        
        for(int i=1;i<=n1;i++)
        {
            for(int j=1;j<=n2;j++)
            {
                if(s1[i-1]==s2[j-1])
                dp[i][j]=1+dp[i-1][j-1];
                
                else
                dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
            }
        }
        
        return dp[n1][n2];
    }
    
```

4.**Print**

```java
    string longestCommonSubsequence(string a) 
    {
       string b=a;
       reverse(b.begin(),b.end());
       
       int n1=a.size();
       int n2=b.size();
       vector<vector<int>> d(n1+1,vector<int>(n2+1,0));
       
       for(int i=1;i<=n1;i++)
       {
           for(int j=1;j<=n2;j++)
           {
               if(a[i-1]==b[j-1])
               {
                   d[i][j]=1+d[i-1][j-1];
               }
               else
               {
                   d[i][j]=max(d[i-1][j],d[i][j-1]);
               }
           }
       }
       
       string v;
       int i=n1,j=n2;
       while(i>0 && j>0)
       {
           if(a[i-1]==b[j-1])
           {
               v.push_back(a[i-1]);
               i--;
               j--;
           }
           
           else
           {
               if(d[i-1][j]>d[i][j-1])
               {
                   i--;
               }
               else
               {
                   j--;
               }
           }
       }
       reverse(v.begin(),v.end());
       return v;
    }
    
```

5.**Gap Method Problems**

General Dp problem which is solved by Gap method

[https://leetcode.com/problems/count-different-palindromic-subsequences/](https://leetcode.com/problems/count-different-palindromic-subsequences/)  
[https://leetcode.com/problems/palindrome-partitioning-ii/](https://leetcode.com/problems/palindrome-partitioning-ii/)  
[https://leetcode.com/problems/minimum-score-triangulation-of-polygon/](https://leetcode.com/problems/minimum-score-triangulation-of-polygon/)

And Leetcode stones problem set are also included.

All the above problems can be solved by gap methodwith minor tweaks.  
Below is a standard code for gap method code.

**count palindromic subsequence**

```java
       int countPalindromicSubsequences(string s) 
        {
          int d[s.length()][s.length()];  
        
          for(int g=0;g<s.length();g++)
          {
              for(int i=0,j=g;j<s.length();i++,j++)
              {
                  if(g==0)
                  {
                      d[i][j]=1;
                  }
                  else if(g==1)
                  {
                      if(s[i]==s[j])
                      {
                          d[i][j]=3;
                      }
                      else
                      {
                          d[i][j]=2;
                      }
                  }
                  
                  else
                  {
                      if(s[i]==s[j])
                      {
                          d[i][j]=d[i][j-1]+d[i+1][j]+1;
                      }
                      else
                      {
                          d[i][j]=d[i][j-1]+d[i+1][j]-d[i+1][j-1];
                      }
                  }
              }
          }
            
            return d[0][s.length()-1];
        }
    
```

6.**Kadans algo**

Identify if problems talks about finding the maximum subarray sum.

[https://leetcode.com/problems/best-time-to-buy-and-sell-stock/](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)  
[https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)  
[https://leetcode.com/problems/arithmetic-slices/](https://leetcode.com/problems/arithmetic-slices/)  
[https://leetcode.com/problems/arithmetic-slices-ii-subsequence/](https://leetcode.com/problems/arithmetic-slices-ii-subsequence/)  
[https://leetcode.com/problems/longest-turbulent-subarray/](https://leetcode.com/problems/longest-turbulent-subarray/)  
[https://leetcode.com/problems/k-concatenation-maximum-sum/](https://leetcode.com/problems/k-concatenation-maximum-sum/)  
[https://leetcode.com/problems/k-concatenation-maximum-sum/](https://leetcode.com/problems/k-concatenation-maximum-sum/)  
[https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/](https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/)  
[https://leetcode.com/problems/ones-and-zeroes/](https://leetcode.com/problems/ones-and-zeroes/)  
[https://leetcode.com/problems/maximum-sum-circular-subarray/](https://leetcode.com/problems/maximum-sum-circular-subarray/)

All the above problems can be solved by gap method with minor tweaks.  
Below is a standard code for gap method code.

```java
    int kad(vector<int> v)
    {
        int c=v[0],o=v[0];
        
        for(int i=1;i<n;i++)
        {
            if(c >= 0)
            {
                c=c+v[i];
            }
            else
            {
                c=v[i];
            }
            
             if(o<c)
            {
                 o=c;
            }
            
        }
        
        return o;
    }

```

7.**Catalan**

Identify if problems talks about counting the number of something.  
eg node,bracket etc.

[https://leetcode.com/problems/unique-binary-search-trees/](https://leetcode.com/problems/unique-binary-search-trees/)

All the above problems can be solved by catalan with minor tweaks.  
Below is a standard code for catalan code.

```java
    int cat(int n)
    {
        int dp[n+1];
        dp[0]=1;
        dp[1]=1;
        for(int i = 2; i < n+1; i++)
        {
            dp[i]=0;
             for(int j = 0; j < i; j++)
             {
                dp[i] += dp[j] * dp[i - 1 - j];
             }
          }
        
        return dp[n];
    }
    
```

Please correct the approach/solution if you find anything wrong.  
And if you like my post then give a thumbs up : ) happy coding

dynamic programming noteshow to solve dpnotes on dpall methodinternplacement

Comments: 21

BestMost VotesNewest to OldestOldest to Newest

Preview

Post

[![aryan_129's avatar](https://assets.leetcode.com/users/aryan_129/avatar_1627145989.png)](/aryan_129)

[aryan\_129](/aryan_129)60

August 3, 2021 9:26 PM

Read More

You can also include the type which requires traversing through the array and then solving subproblems in left side and right side. Eg - Matrix Chain Multiplication, Egg Dropping problems etc.  
Some examples from leetcode:

1.  [https://leetcode.com/problems/burst-balloons/](https://leetcode.com/problems/burst-balloons/)
2.  [https://leetcode.com/problems/scramble-string/](https://leetcode.com/problems/scramble-string/)
3.  [https://leetcode.com/problems/parsing-a-boolean-expression/](https://leetcode.com/problems/parsing-a-boolean-expression/)

6

Show 1 reply

Reply

Share

Report

[![Nam_22's avatar](https://assets.leetcode.com/users/nam_22/avatar_1586063023.png)](/Nam_22)

[Nam\_22](/Nam_22)107

August 3, 2021 7:13 PM

Read More

You could also add following to  
Longest Increasing Subsequence (LIS) as well.

1.  [https://leetcode.com/problems/russian-doll-envelopes/](https://leetcode.com/problems/russian-doll-envelopes/)
2.  [https://leetcode.com/problems/maximum-height-by-stacking-cuboids/description/](https://leetcode.com/problems/maximum-height-by-stacking-cuboids/description/)

4

Show 2 replies

Reply

Share

Report

[![Mazhar_MIK's avatar](https://assets.leetcode.com/users/mkhan31995/avatar_1588411556.png)](/Mazhar_MIK)

[Mazhar\_MIK](/Mazhar_MIK)![](/static/images/badges/dcc-2021-6.png)645

3 days ago

Read More

One Spot Qns for DP : [https://github.com/MAZHARMIK/Interview\_DS\_Algo/tree/master/DP](https://github.com/MAZHARMIK/Interview_DS_Algo/tree/master/DP)

3

Reply

Share

Report

[![CarlGWatts's avatar](https://www.gravatar.com/avatar/31cb3cf94a0c14a2a27a618bedf78d27.png?s=200)](/CarlGWatts)

[CarlGWatts](/CarlGWatts)2

8 hours ago

Read More

[@vchand324](https://leetcode.com/vchand324) The code in the first three functions you define refer to undefined variables "n" and "w" and declare a two-dimensional array of "bool" and then store "int"s in the array. These functions couldn't possibly compile, let alone work. You shouldn't write an article contain code examples that couldn't even compile, let alone work. Also your variable names do nothing to help the reader understand what they are supposed to represent. Also your code contains if-else statements of the form "if a<b then ... else if a>=b then ..." which is confusing code that can be replaced with an if statements of the form "if a<b then ... else ...". You should fix all these code examples and make sure the code can compile and run.

2

Show 1 reply

Reply

Share

Report

[![frostbyte4916's avatar](https://www.gravatar.com/avatar/b8af42b2b48d119e404a2d2e43b0a52f.png?s=200)](/frostbyte4916)

[frostbyte4916](/frostbyte4916)1

20 hours ago

Read More

It's such an amazing article. Thank you for that.  
Seems there is an error in the else if statement of 01 knapsack code. I think it should be dp\[i-1\]\[j - nums\[i-1\]\]. I could be wrong, in which case please correct me.

1

Show 1 reply

Reply

Share

Report

[![hostingshoutcas's avatar](https://www.gravatar.com/avatar/f217321e847e93fb1f4dd5d902741025.png?s=200)](/hostingshoutcas)

[hostingshoutcas](/hostingshoutcas)1

Last Edit: 18 hours ago

Read More

Thanks for your effort. It will contribute a lot in my recent project of my [japanese](h
ttps://japan-beyond.com) client.

1

Show 1 reply

Reply

Share

Report

[![prashanth180's avatar](https://assets.leetcode.com/users/user3823u/avatar_1617721892.png)](/prashanth180)

[prashanth180](/prashanth180)2

2 days ago

Read More

Nice

1

Reply

Share

Report

[![boppanasusanth's avatar](https://assets.leetcode.com/users/boppanasusanth/avatar_1627898000.png)](/boppanasusanth)

[boppanasusanth](/boppanasusanth)![](/static/images/badges/dcc-2021-7.png)26

Last Edit: 2 days ago

Read More

[@vchand324](https://leetcode.com/vchand324) In your profile, note that Indiana Institute of Technology and Indian Institute of Technology are different. Good list by the way.

1

Show 1 reply

Reply

Share

Report

[![Coder_Shubham_24's avatar](https://assets.leetcode.com/users/tech_runner/avatar_1589173665.png)](/Coder_Shubham_24)

[Coder\_Shubham\_24](/Coder_Shubham_24)![](/static/images/badges/dcc-2021-5.png)49

2 days ago

Read More

helpful...thanks for the effort and contribution .. means a lot to us..

1

Reply

Share

Report

[![banerjee_abir's avatar](https://assets.leetcode.com/users/banerjee_abir/avatar_1627379344.png)](/banerjee_abir)

[banerjee\_abir](/banerjee_abir)2

3 days ago

Read More

really helpful.  
Thanks

