import nolds
from nolds.measures import rowwise_euclidean
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

st.title('Nonlinear Dynamical System Analysis')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -300px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Menu')

app_mode = st.sidebar.selectbox('Please Select',
                                ['About', 'Sample Dataset', 'Help', 'View Data', 'Descriptive Statistics'
                                 , 'System Dynamics']
                                )

if app_mode == 'About':
    st.markdown('''
    This webpage is an application for evaluating Nonlinear Dynamical System (NDS) behavior from the system's output time series.
    Various NDS quantifiers can be measured using this application. \n
    Please note that you have considered the assumption of chaotic behavior for the system when using NDS quantifiers. \n
    Feel free to use this web application for educational purposes. In this case, Please cite this website in your publications. \n
    The author does not have any responsibility for the consequences of using the obtained results. \n  
    This web application is updated continuously and will provide new features for time series analyses. 
    Do not forget to check it out for new features.  
    ''')

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -300px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.image('lorenz_logo.png')

    st.markdown('''
        Author : Hamed Ghomshchi, Ph.D. \n
        Email : h_ghomashchi@yahoo.com 
        ''')

elif app_mode == 'Sample Dataset':

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -300px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    df = pd.read_excel('lorenz_data.xls')
    # print(df)
    # pr = ProfileReport(df, explorative=True)
    st.header('**Lorenz Attractor**')
    st.write(df)
    st.write('---')
    # st.header('**Pandas Profiling Report**')
    # st_profile_report(pr)

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(3, 1, 1)
    ax.plot(
        df["x"]
    )
    ax = fig1.add_subplot(3, 1, 2)
    ax.plot(
        df["y"]
    )
    ax = fig1.add_subplot(3, 1, 3)
    ax.plot(
        df["z"]
    )
    ax.set_xlabel("State Variables")
    st.write(fig1)

    fig2 = plt.figure(2)
    ax = fig2.gca(projection="3d")
    ax.plot(df["x"], df["y"], df["z"], 'gray')
    st.write(fig2)

elif app_mode == 'Help':

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -300px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.markdown('''
            This web application calculates different measures for analyzing the behavior of nonlinear dynamical systems (NDS) 
            from the system's outputs. \n
            In addition to descriptive statistics of the time series, this web application calculates the following 
            NDS quantifiers: \n
            -   Correlation Dimension (Dc) \n
            -   Appropriate number of Embedding Dimension for the reconstruction of state space (d) \n
            -   Appropriate reconstruction delay (Lag) using Auto Correlation Function \n
            -   Largest Lyapunov Exponent (LyE) using Rosenstein algorithm \n
            \n  
            For more information about the procedure, please refer to: \n
            -   Chaos and Nonlinear Dynamics: An Introduction for Scientists and Engineers, Robert C. Hilborn \n
            -   Innovative Analyses of Human Movement, Nicholas Stergiou \n
            ''')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left: -300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


####################################################################
####################################################################

elif app_mode == 'View Data':

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -300px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.markdown(' ## Output')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left: -300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader("Upload Datafile", type=['xls'])

    if not uploaded_file:
        df = pd.read_excel('lorenz_data.xls')
    else:
        df = pd.read_excel(uploaded_file)

    kpi1, kpi2 = st.columns(2)

    with kpi1:
        kpi1 = st.markdown("**No of Columns**")
        kpi1_text = st.markdown(df.shape[1])

    with kpi2:
        st.markdown("**No of Rows**")
        kpi2_text = st.markdown(df.shape[0])

    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')

    filtered = st.multiselect("Show Columns", options=list(df.columns))
    st.write(df[filtered])
    st.line_chart(df[filtered])


elif app_mode == 'Descriptive Statistics':

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -300px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.markdown(' ## Descriptive Statistics')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left: -300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        kpi1 = st.markdown(f"<h1 style='text-align: center; color: black;'>Mean</h1>",
                           unsafe_allow_html=True)
        kpi1_text = st.markdown(f"<h1 style='text-align: center; color: black;'>{0}</h1>",
                                    unsafe_allow_html=True)

    with kpi2:
        kpi2 = st.markdown(f"<h1 style='text-align: center; color: black;'>STD</h1>",
                           unsafe_allow_html=True)
        kpi2_text = st.markdown(f"<h1 style='text-align: center; color: black;'>{0}</h1>",
                                unsafe_allow_html=True)

    with kpi3:
        kpi3 = st.markdown(f"<h1 style='text-align: center; color: black;'>Range</h1>",
                           unsafe_allow_html=True)
        kpi3_text = st.markdown(f"<h1 style='text-align: center; color: black;'>{0}</h1>",
                                unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Upload Datafile", type=['xls'])

    if not uploaded_file:
        df = pd.read_excel('lorenz_data.xls')
    else:
        df = pd.read_excel(uploaded_file)

    filtered = st.multiselect("Select Only One Columns", options=list(df.columns))
    data = pd.DataFrame(df[filtered]).to_numpy()

    try:
        Mean = np.mean(data)
        STD = np.std(data)
        Range = np.ptp(data)

        st.write('---')
        st.line_chart(data)

        kpi1_text.write(f"<h1 style='text-align: center; color: black;'>{round(Mean, 2)}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: black;'>{round(STD, 2)}</h1>", unsafe_allow_html=True)
        kpi3_text.write(f"<h1 style='text-align: center; color: black;'>{round(Range, 2)}</h1>", unsafe_allow_html=True)
    except:
        pass

elif app_mode == 'System Dynamics':

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -300px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left: -300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader("Upload Datafile", type=['xls'])

    if not uploaded_file:
        df = pd.read_excel('lorenz_data.xls')
    else:
        df = pd.read_excel(uploaded_file)

    filtered = st.multiselect("Select Only One Columns", options=list(df.columns))
    data_nf = pd.DataFrame(df[filtered]).to_numpy() # Not flatten data
    data =[] # Flatten data
    for item in data_nf:
        data.extend(item)
    data = np.array(data)

    quantifiers = st.sidebar.selectbox('Please Select',
                                    ['Correlation Dimension','Lag Calculation','Lyapunov Exponent']
                                    )

    if quantifiers == 'Correlation Dimension':

        st.markdown(
            """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left: -300px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
        st.markdown(' ## Corrlation Dimension')
        corrDimList = []

        kpi1, kpi2 = st.columns(2)

        with kpi1:
            kpi1 = st.markdown(f"<h1 style='text-align: center; color: black;'>d</h1>",
                               unsafe_allow_html=True)
            kpi1_text = st.markdown(f"<h1 style='text-align: center; color: black;'>{0}</h1>",
                                    unsafe_allow_html=True)

        with kpi2:
            kpi2 = st.markdown(f"<h1 style='text-align: center; color: black;'>Dc</h1>",
                               unsafe_allow_html=True)
            kpi2_text = st.markdown(f"<h1 style='text-align: center; color: black;'>{0}</h1>",
                                    unsafe_allow_html=True)
        st.write('---')

        try:
            st.sidebar.markdown('---')
            dimension = st.sidebar.slider('Embedding Dimension', min_value=1, max_value=30, value=25, step=1)
            st.sidebar.markdown('---')

            for d in range(1, dimension+1):
                Dc = nolds.corr_dim(data, d, rvals=None,
                                   dist=rowwise_euclidean, fit=u'RANSAC', debug_plot=False,
                                   debug_data=False, plot_file=None)

                kpi1_text.write(f"<h1 style='text-align: center; color: black;'>{d}</h1>",
                                unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: center; color: black;'>{round(Dc, 2)}</h1>",
                                unsafe_allow_html=True)
                corrDimList.append(Dc)

            fig = plt.figure()
            plt.plot(corrDimList)
            plt.xlabel("Embedding Dimension ")
            plt.ylabel("Correlation Dimension")
            st.write(fig)
            for i in range(dimension+1):
                st.write(i+1,round(corrDimList[i],2))
        except:
            pass

    if quantifiers == 'Lag Calculation':

        st.markdown(
            """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left: -300px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
        st.markdown(' ## Auto Correlation Function')

        kpi1 = st.markdown(f"<h1 style='text-align: center; color: black;'>Lag</h1>",
                            unsafe_allow_html=True)
        kpi1_text = st.markdown(f"<h1 style='text-align: center; color: black;'>{0}</h1>",
                            unsafe_allow_html=True)
        st.write('---')

        try:
            autoCorr = signal.correlate(data,data)
            #
            fig = plt.figure()
            plt.plot(autoCorr)#[0:len(autoCorr)//10])
            plt.xlabel("Lag")
            plt.ylabel("Correlation")
            st.write(fig)

            minimums = signal.argrelextrema(data, np.less)
            kpi1_text.write(f"<h1 style='text-align: center; color: black;'>{minimums[0][0]}</h1>",
                            unsafe_allow_html=True)
        except:
            pass

    if quantifiers == 'Lyapunov Exponent':

        st.markdown(
            """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left: -300px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        kpi1 = st.markdown(f"<h1 style='text-align: center; color: black;'>Lyapunov Exponent</h1>",
                           unsafe_allow_html=True)
        kpi1_text = st.markdown(f"<h1 style='text-align: center; color: black;'>{0}</h1>",
                                unsafe_allow_html=True)

        try:
            d = st.text_input("Enter No. Of Embedding Dimensions", "3")
            delay = st.text_input("Enter Time Delay", "None")

            if delay != "None":
                delay = int(delay)
                lyaExp = nolds.lyap_r(data, emb_dim=int(d), lag=delay, min_tsep=None, tau=1, min_neighbors=10, trajectory_len=20,
                     fit=u'RANSAC', debug_plot=False, debug_data=False, plot_file=None, fit_offset=0)
                kpi1_text.write(f"<h1 style='text-align: center; color: black;'>{round(lyaExp,3)}</h1>",
                                unsafe_allow_html=True)
            else:
                lyaExp = nolds.lyap_r(data, emb_dim=int(d), lag=None, min_tsep=None, tau=1, min_neighbors=10,
                                      trajectory_len=20,
                                      fit=u'RANSAC', debug_plot=False, debug_data=False, plot_file=None, fit_offset=0)
                kpi1_text.write(f"<h1 style='text-align: center; color: black;'>{round(lyaExp,3)}</h1>",
                                unsafe_allow_html=True)
        except:
            pass
