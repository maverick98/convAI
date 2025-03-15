import streamlit as st
from rag_pipeline import retrieve_answer
from guardrails import validate_query, filter_response

st.title("📊 Boeing Financial Report Assistant")
st.markdown("Ask questions about Boeing's 2022 & 2023 financial reports!")

query = st.text_input("🔍 Enter your financial question:")

if st.button("Search"):
    if query:
        # Validate user query (Guardrail)
        validation_status, validation_msg = validate_query(query)
        if not validation_status:
            st.error(f"🚫 {validation_msg}")
        else:
            with st.spinner("Searching..."):
                answer, confidence = retrieve_answer(query)
                filtered_answer = filter_response(answer, confidence)
            
            st.success("✅ Answer Retrieved!")
            st.write("**Query:**", query)
            st.write("**Answer:**", filtered_answer)
            st.write(f"**Confidence Score:** {confidence:.2f}")  # Display confidence

    else:
        st.warning("⚠️ Please enter a query!")
