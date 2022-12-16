import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main() :
    st.title('K-Means 클러스터링')

    # 1. csv 파일을 업로드 할 수 있다.
    file = st.file_uploader('CSV파일 업로드', type=['csv'])
   
    if file is not None :
        # 2. csv 파일은, 판다스로 읽어 화면에 표시한다.
        df = pd.read_csv(file)
        st.dataframe( df )
        # 3.X로 사용할 컬럼을 설정할 수 있다.
        selected_columns = st.multiselect('X로 사용할 컬럼을 선택하세요', df.columns)
        if len(selected_columns) != 0 :
            X = df[selected_columns]
            st.dataframe(X)

            st.subheader('WCSS를 위한 클러스터링 개수 선택')
            max_number = st.slider('최대 그룹 수 선택', 2, 20, value=10) # value = 유저가 선택 안했을 때 디폴트 값...
        
            wcss = []
            for k in np.arange(1, max_number+1) :
                kmeans = KMeans(n_clusters= k, random_state=5)
                kmeans.fit(X)
                wcss.append( kmeans.inertia_ )

            # st.write(wcss)  #이렇게 디버깅 해놓고, 정상 확인했으면 주석 처리

            fig1 = plt.figure()
            x = np.arange(1, max_number+1)
            plt.plot( x, wcss )
            plt.title('The Elbow Method')
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS')
            st.pyplot(fig1)

        # 실제로 그룹핑할 개수 선택!
        # K = st.slider('그룹 개수 결정', 1, max_number) # 해보니까 로딩 시간이 많이 걸리더라..
            K = st.number_input('그룹 개수 결정', 1, max_number)

            kmeans = KMeans(n_clusters= K, random_state=5)

            y_pred = kmeans.fit_predict(X)

            df['Group'] = y_pred

            st.dataframe( df.sort_values('Group') )

        # 데이터프레임은 이렇게 바로 파일로 저장하는 함수가 있음...
            df.to_csv('result.csv')

if __name__ == '__main__' :
    main()