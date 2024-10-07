import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    data = pd.read_csv('student-scores.csv')
    return data

# Fungsi untuk melakukan regresi linear
def perform_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2, X_test, y_test, y_pred

# Aplikasi Streamlit
def main():
    st.title('Analisis Skor Mahasiswa')

    # Memuat data
    data = load_data()

    # Menampilkan dataset
    st.subheader('Dataset')
    st.write(data.head())

    # Memilih variabel independen dan dependen
    st.subheader('Pilih Variabel')
    target = st.selectbox('Pilih variabel target (Y):', data.columns)
    features = st.multiselect('Pilih variabel fitur (X):', [col for col in data.columns if col != target])

    if not features:
        st.warning('Silakan pilih setidaknya satu variabel fitur.')
        return

    X = data[features]
    y = data[target]

    # Melakukan regresi linear
    model, mse, r2, X_test, y_test, y_pred = perform_regression(X, y)

    # Menampilkan hasil
    st.subheader('Hasil Regresi Linear')
    st.write(f'Mean Squared Error: {mse:.4f}')
    st.write(f'R-squared Score: {r2:.4f}')

    # Menampilkan koefisien
    st.subheader('Koefisien Model')
    coef_df = pd.DataFrame({'Variabel': X.columns, 'Koefisien': model.coef_})
    st.write(coef_df)

    # Visualisasi hasil prediksi vs aktual
    st.subheader('Visualisasi Prediksi vs Aktual')
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Nilai Aktual')
    ax.set_ylabel('Nilai Prediksi')
    ax.set_title('Prediksi vs Aktual')
    st.pyplot(fig)

if __name__ == '__main__':
    main()