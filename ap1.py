import streamlit as st
import pickle
import numpy as np
import sklearn  # <--- Add this specifically

# 1. Load your saved model and scaler
try:
    model = pickle.load(open('trained_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
except FileNotFoundError:
    st.error("Model or Scaler files not found! Please run the saving cells first.")

def main():
    st.title("Diabetes Prediction App")
    st.write("Enter the following details to check the diabetes status:")

    # 2. UI Layout
    col1, col2 = st.columns(2)

    with col1:
        preg = st.text_input('Number of Pregnancies', value="0")
        gluc = st.text_input('Glucose Level', value="0")
        bp = st.text_input('Blood Pressure value', value="0")
        skin = st.text_input('Skin Thickness value', value="0")

    with col2:
        ins = st.text_input('Insulin Level', value="0")
        bmi = st.text_input('BMI value', value="0")
        dpf = st.text_input('Diabetes Pedigree Function', value="0")
        age = st.text_input('Age', value="0")

    # 3. Prediction Logic
    if st.button("Predict Result"):
        user_input = [float(preg), float(gluc), float(bp), float(skin),
                      float(ins), float(bmi), float(dpf), float(age)]

        input_data = np.asarray(user_input).reshape(1,-1)
        std_data = scaler.transform(input_data)
        prediction = model.predict(std_data)

        st.markdown("---") # Visual separator

        if prediction[0] == 1:
            st.error("### Result: Positive")
            st.write("The person is likely Diabetic. Please consult a doctor for a professional diagnosis.")

            # --- PRECAUTIONS & DIET SECTION ---
            st.subheader("📋 Recommended Precautions & Diet")

            col_diet, col_links = st.columns(2)

            with col_diet:
                st.markdown("""
                **Dietary Tips:**
                * **Eat more Fiber:** Focus on whole grains, beans, and leafy greens.
                * **Reduce Sugars:** Avoid sodas, candies, and processed snacks.
                * **Portion Control:** Use smaller plates to manage calorie intake.
                * **Healthy Fats:** Opt for nuts, seeds, and olive oil.
                """)

            with col_links:
                st.markdown("""
                **Educational Resources:**
                * [Understanding Diabetes (Mayo Clinic)](https://www.youtube.com/watch?v=X9ivR4y03DE)
                * [Best Foods for Diabetics](https://www.youtube.com/watch?v=PrUu8V5A1iI)
                * [Exercise Tips for Blood Sugar](https://www.youtube.com/watch?v=X_pU6H7P01Q)
                """)

        else:
            st.success("### Result: Negative")
            st.write("The person is likely Not Diabetic. Continue maintaining a healthy lifestyle!")

if __name__ == '__main__':
    main()
