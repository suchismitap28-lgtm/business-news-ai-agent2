with st.spinner("Generating structured, descriptive answers..."):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior equity research analyst writing a professional investor brief. "
                "For each question, write a separate numbered section in this format:\n\n"
                "### <Question Title>\n"
                "<div class='answer-block'>\n"
                "<b>Summary:</b> one strong sentence.\n"
                "<br>\n"
                "Then 3–5 descriptive lines with insights, implications, and data if available.\n"
                "<br>\n"
                "End with **Sources:** followed by markdown links.\n"
                "</div>\n\n"
                "Use HTML and markdown formatting for clarity, line breaks, and spacing. "
                "Never combine multiple answers in one paragraph."
            )
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n"
                f"Questions:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "Provide well-formatted HTML/Markdown for each numbered answer."
            )
        }
    ]

    try:
        ans = openai_chat(messages, max_tokens=1500)
    except Exception:
        ans = hf_answer(question, context)

# --- Post-formatting cleanup to force line breaks ---
ans = ans.replace("1.", "#### **1️⃣ ").replace("2.", "#### **2️⃣ ") \
         .replace("3.", "#### **3️⃣ ").replace("4.", "#### **4️⃣ ") \
         .replace("5.", "#### **5️⃣ ").replace("6.", "#### **6️⃣ ")
ans = ans.replace("</div>", "</div><br>")

st.markdown("### 💡 Analytical Answers")
st.markdown(ans, unsafe_allow_html=True)
st.success("✅ Done — structured insights generated!")
