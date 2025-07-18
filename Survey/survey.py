import streamlit as st
import re
from datetime import datetime

st.set_page_config(page_title= "نبض الخليج 💓",layout="centered")

st.markdown("<h1 style='text-align: right;'>نبض الخليج 💓</h1>", unsafe_allow_html=True)
st.markdown("Thank you for using our smart surveillance system. Please take a moment to give us your feedback.")

# Use session_state to retain values after submission
if "form_data" not in st.session_state:
    st.session_state.form_data = {}

with st.form("feedback_form"):
    st.subheader("📢 Alerts Review")
    helpful = st.checkbox("✅ Were the alerts we sent helpful?", value=st.session_state.form_data.get("helpful", False))
    false_alarm = st.checkbox("❌ Some alerts were false alarms", value=st.session_state.form_data.get("false_alarm", False))
    minor_alarm = st.checkbox("⚠️ Some alerts were minor ('Good but could be better')", value=st.session_state.form_data.get("minor_alarm", False))

    st.subheader("📲 Usage")
    used_notifications = st.checkbox("Have you used any of our real-time notifications?", value=st.session_state.form_data.get("used_notifications", False))

    st.subheader("💬 Your Feedback")
    suggestions = st.text_area("Tell us how we can serve you better:", value=st.session_state.form_data.get("suggestions", ""))

    st.subheader("📞 Support Options")
    contact_option = st.selectbox(
        "Would you like to connect with our support team?",
        ["No, thank you", "Yes, via phone call", "Yes, connect me to a team member"],
        index=["No, thank you", "Yes, via phone call", "Yes, connect me to a team member"].index(
            st.session_state.form_data.get("contact_option", "No, thank you"))
    )

    contact_info = {}
    email = ""
    country_code = ""
    phone_number = ""

    if contact_option == "Yes, connect me to a team member":
        email = st.text_input("📧 Enter your email address", value=st.session_state.form_data.get("email", ""))
    elif contact_option == "Yes, via phone call":
        country_code = st.text_input("🌍 Country Code (e.g., +1, +20)", value=st.session_state.form_data.get("country_code", ""))
        phone_number = st.text_input("📞 Phone Number", value=st.session_state.form_data.get("phone_number", ""))

    submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        # Save form state in session to retain values
        st.session_state.form_data = {
            "helpful": helpful,
            "false_alarm": false_alarm,
            "minor_alarm": minor_alarm,
            "used_notifications": used_notifications,
            "suggestions": suggestions,
            "contact_option": contact_option,
            "email": email,
            "country_code": country_code,
            "phone_number": phone_number
        }

        # Validation checks
        valid = True
        if contact_option == "Yes, connect me to a team member":
            if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.error("❌ Please enter a valid email address.")
                valid = False
            else:
                contact_info["Email"] = email

        elif contact_option == "Yes, via phone call":
            if not country_code or not phone_number:
                st.error("❌ Please enter both country code and phone number.")
                valid = False
            else:
                contact_info["Phone"] = f"{country_code}{phone_number}"

        if valid:
            st.success("✅ Thank you for your feedback!")

            st.write("Your responses:")
            st.json({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Helpful Alerts": helpful,
                "False Alarms": false_alarm,
                "Minor Alarms": minor_alarm,
                "Used Notifications": used_notifications,
                "Suggestions": suggestions,
                "Contact Option": contact_option,
                **contact_info
            })
