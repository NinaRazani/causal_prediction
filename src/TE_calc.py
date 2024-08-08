import pandas as pd 
import numpy as np 
import numba 
from numba import jit
import time
# import progressbar
from progress.bar import Bar  


@jit
def TE(x,y,pieces,j):
    d_x=np.zeros((j,4))
    sit=len(x)
    # temp1=list(range(sit-1))
    # random.seed(101)
    # np.random.shuffle(temp1)    #shuffle indices of input dataframe
    # select=np.array(temp1[:j])

    # d_x[:,0]=x[select+1]
    # d_x[:,1]=x[select]
    # d_x[:,2]=y[select+1]
    # d_x[:,3]=y[select]

    d_x[:,0] = x[1:sit]   # x lagged by 1 step
    d_x[:,1] = x[0:sit-1]
    d_x[:,2] = y[1:sit]   # y lagged by 1 step
    d_x[:,3] = y[0:sit-1]



    x_max=np.max(x); x_min=np.min(x); y_max=np.max(y); y_min=np.min(y)

    delta1=(x_max-x_min)/(2*pieces); delta2=(y_max-y_min)/(2*pieces)

    L1=np.linspace(x_min+delta1,x_max-delta1,pieces); L2=np.linspace(y_min+delta2,y_max-delta2,pieces)
    #divide the variable space into equal intervals based on the specified number of pieces.bin intervals

    dist1=np.zeros((pieces,2))  #initialized to store the counts of points falling within each bin in the x and y dimensions
    count=-1

    for q1 in range(pieces):   #loop on number of bins
        k1=L1[q1]; k2=L2[q1]
        count+=1
        count1=0;count2=0   #the number of points that place in each bins of x (count 1), place in each bins of y(count2)
        for i in range(j):     #loop on number of rows of fixed dataframe (the number of rows considered for calculate TE)
            if d_x[i,1]>=(k1-delta1) and d_x[i,1]<=(k1+delta1):  #the bin defined by (k1-delta1) and (k1+delta1)
                count1+=1
            if d_x[i,3]>=(k2-delta2) and d_x[i,3]<=(k2+delta2):
                count2+=1
        #print(dist1)
        #print(count1)
        dist1[count,0]=count1; dist1[count,1]=count2  #store number of points of each bin in correspond index in dist1

    dist1[:,0]=dist1[:,0]/sum(dist1[:,0]); dist1[:,1]=dist1[:,1]/sum(dist1[:,1])  #normalize to obtain probability distribution

    dist2=np.zeros((pieces,pieces,3))
    for q1 in range(pieces):
        for q2 in range(pieces):
            # print('222')
            k1=L1[q1]; k2=L1[q2]
            k3=L2[q1]; k4=L2[q2]
            count1=0;count2=0;count3=0
            for i1 in range(j):
                    if d_x[i1,0]>=(k1-delta1) and d_x[i1,0]<=(k1+delta1) and d_x[i1,1]>=(k2-delta1) and d_x[i1,1]<=(k2+delta1):  #calculate the existence of consecutive points in bins x time serie
                        count1=count1+1;

                    if d_x[i1,2]>=(k3-delta2) and d_x[i1,2]<=(k3+delta2) and d_x[i1,3]>=(k4-delta2) and d_x[i1,3]<=(k4+delta2):  #calculate the existence of consecutive points in bins y time serie
                        count2=count2+1;

                    if d_x[i1,1]>=(k1-delta1) and d_x[i1,1]<=(k1+delta1) and d_x[i1,3]>=(k4-delta2) and d_x[i1,3]<=(k4+delta2):  #calculate the existence of x(i1), y(i1) points in bin x and bin y simulatuasly
                        count3=count3+1;

            dist2[q1,q2,0]=count1; dist2[q1,q2,1]=count2; dist2[q1,q2,2]=count3;

    dist2[:,:,0]=dist2[:,:,0]/np.sum(dist2[:,:,0])
    dist2[:,:,1]=dist2[:,:,1]/np.sum(dist2[:,:,1])
    dist2[:,:,2]=dist2[:,:,2]/np.sum(dist2[:,:,2])

    dist3=np.zeros((pieces,pieces,pieces,2));

    for q1 in range(pieces):
        for q2 in range(pieces):
            for q3 in range(pieces):
                k1=L1[q1];k2=L1[q2];k3=L1[q3]
                k4=L2[q1];k5=L2[q2];k6=L2[q3]
                count1=0;count2=0
                for i1 in range(j):
                    if d_x[i1,0]>=(k1-delta1) and d_x[i1,0]<=(k1+delta1) and d_x[i1,1]>=(k2-delta1) and d_x[i1,1]<=(k2+delta1) and d_x[i1,3]>=(k6-delta2) and d_x[i1,3]<=(k6+delta2):
                        count1=count1+1
                        # two consecutive x and a y

                    if d_x[i1,2]>=(k4-delta2) and d_x[i1,2]<=(k4+delta2) and d_x[i1,3]>=(k5-delta2) and d_x[i1,3]<=(k5+delta2) and d_x[i1,1]>=(k3-delta1) and d_x[i1,1]<=(k3+delta1):
                        count2=count2+1
                        # two consecutive y and a x

                dist3[q1,q2,q3,0]=count1; dist3[q1,q2,q3,1]=count2;

    dist3[:,:,:,0]=dist3[:,:,:,0]/np.sum(dist3[:,:,:,0]); dist3[:,:,:,1]=dist3[:,:,:,1]/np.sum(dist3[:,:,:,1]);

    sum_f_1=0;sum_f_2=0
    for k1 in range(pieces):
        for k2 in range(pieces):
            if dist2[k1,k2,1]!=0 and dist1[k2,1]!=0:
                sum_f_1=sum_f_1-dist2[k1,k2,1]*np.log2(dist2[k1,k2,1]/dist1[k2,1])  # sum = sum - p(y1,y2))*log(p(y1,y2)/p(y))

            if dist2[k1,k2,0]!=0 and dist1[k2,0]!=0:
                sum_f_2=sum_f_2-dist2[k1,k2,0]*np.log2(dist2[k1,k2,0]/dist1[k2,0])

    sum_s_1=0;sum_s_2=0
    for k1 in range(pieces):
        for k2 in range(pieces):
            for k3 in range(pieces):
                if dist3[k1,k2,k3,1]!=0 and dist2[k3,k2,2]!=0:
                    sum_s_1=sum_s_1-dist3[k1,k2,k3,1]*np.log2(dist3[k1,k2,k3,1]/dist2[k3,k2,2]) # sum = sum - p(y1, y2, x)*log(p(y1, y2, x)/p(x, y))

                if dist3[k1,k2,k3,0]!=0 and dist2[k2,k3,2]!=0:
                    sum_s_2=sum_s_2-dist3[k1,k2,k3,0]*np.log2(dist3[k1,k2,k3,0]/dist2[k2,k3,2])

    en_1_2=sum_f_1-sum_s_1
    en_2_1=sum_f_2-sum_s_2

    return en_1_2, en_2_1




def comp_TE(ts):
  # L = 0.8*ts.shape[0]
  L = ts.shape[0]
  A = np.zeros((8,8))
  L = int(L)
  L1 = L
  t = 0
  bar = Bar('Processing', max=20, fill='@', suffix='%(percent)d%%')
  for i in range(ts.shape[1]):
    for j in range(i+1,ts.shape[1]):
      t += 1
      # print('     ',t/10.03,'%\r')
    #   time.sleep(0.0000001)
      bar.next()
      te1,te2 = TE(ts[:L,i],ts[:L,j], 10, L1-1)     #call transfer entropy

      A[i,j] = te1
      A[j,i] = te2
      # if te1 >= te2:
      #   A[i,j] = te1-te2
      # if te1 < te2:
      #   A[j,i] = te2-te1
  bar.finish()
  return A


def comp_avg_TE(df, i):
  #df is a 96 size dataframe and i is a column that g calculate on it
  A = comp_TE(np.array(df))
  # out_flow = np.mean(A[i, :])
  # in_flow = np.mean(A[:, i])
  in_flow = np.mean(A, axis=1)[i]  # row-wise
  out_flow = np.mean(A, axis=0)[i]  # column-wise

  return in_flow, out_flow


def create_traintarg(df, wind_size):
    # calc_df = pd.DataFrame(columns=df.columns)
  data = []
  for j in range(0, len(df)-(wind_size-1), 1):   
    in_f=[]    #in_flows of a column stored here
    out_f =[]  #out_flow of a colunm stored here

    # start_index = j
    # end_index = j + 96
    # current_96_data = df.iloc[j: j+96, :]
    for k in range(len(df.columns)): 
      in_flow = 0
      out_flow = 0
      # Apply the desired function on current 96 data points
      in_flow, out_flow = comp_avg_TE(df.iloc[j: j+wind_size, :], k)
      in_f.append(in_flow)
      out_f.append(out_flow)

    # ret_4 = 0
    # next_4_data = df.iloc[j + 96:j + 96 + 4, :]
    # if len(next_4_data)>3:
    #   ret_4 = calculate_metric(next_4_data)

    data.append(in_f)
    data.append(out_f)
    print('\n','done one window')
    # data.append(ret_4)
  return data



def calculate_metric(chunk):
  # Select the first and fourth elements of the chunk
  first_element = chunk.iloc[0]
  fourth_element = chunk.iloc[3]

    # Calculate the difference and divide by the first element
  metric = (fourth_element - first_element) / first_element

  return metric
