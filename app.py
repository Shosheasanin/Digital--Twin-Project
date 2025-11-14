import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

#  settings 
CSV = "Data/sensor_data.csv"       
WIN, ZTHR = 5, 1.0                 
REFRESH_MS = 60_000                

#  auto refresh 
st_autorefresh(interval=REFRESH_MS, key="tick")
st.title("🌊 mini river dashboard")

# data 
df = pd.read_csv(CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
dfw = df.pivot_table(index="timestamp", columns=["station_id","variable"], values="value").sort_index(axis=1)

# anomaly helpers 
def flag_anom(s, w=WIN, zthr=ZTHR):
    m = s.rolling(w, 1).mean()
    sd = s.rolling(w, 1).std().replace(0, np.nan)
    z = (s - m) / sd
    return (z > zthr).fillna(False), z

# all anomalies table 
rows=[]
for (stn,var) in dfw.columns:
    s = dfw[(stn,var)].dropna()
    f, z = flag_anom(s)
    for ts,val in s[f].items():
        rows.append({"timestamp":ts,"station_id":stn,"variable":var,"value":float(val),"z":float(z.loc[ts])})
hits = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

# header cards 
stations = sorted(df["station_id"].unique())
variables = sorted(df["variable"].unique())
stn = st.selectbox("station", stations, index=0, key="stn")
var = st.selectbox("variable", variables, index=0, key="var")

col1, col2 = st.columns(2)
if not hits.empty:
    last = hits.iloc[-1]
    col1.metric("latest alert", f"{last.station_id} · {last.variable}", last.timestamp.strftime("%Y-%m-%d %H:%M UTC"))
else:
    col1.metric("latest alert", "none", "—")

sel = dfw[(stn, var)].dropna()
if len(sel) >= 3:
    slope = np.polyfit(range(3), sel.tail(3).values, 1)[0]
    col2.metric("trend", f"{'↑' if slope>0 else ('↓' if slope<0 else '→')} {slope:.2f}/step")
else:
    col2.metric("trend", "not enough points")

#  chart with anomaly dots 
flags, _ = flag_anom(sel)
fig = plt.figure(figsize=(10,4))
plt.plot(sel.index, sel.values, label=f"{stn} {var}")
plt.scatter(sel.index[flags], sel[flags], s=70, label="anomaly")
plt.xlabel("time"); plt.ylabel(var); plt.grid(True); plt.legend(); plt.tight_layout()
st.pyplot(fig)

st.subheader("recent alerts")
st.dataframe(hits.tail(10) if not hits.empty else pd.DataFrame(columns=["timestamp","station_id","variable","value","z"]))

#  alerts (email)


def send_email(subject, body):
    try:
        import smtplib, ssl
        from email.message import EmailMessage

        e = st.secrets["email"]  

        msg = EmailMessage()
        msg["From"] = e["user"]          
        msg["To"] = e["to"]
        msg["Subject"] = subject
        msg.set_content(body)

        ctx = ssl.create_default_context()
        with smtplib.SMTP(e["smtp"], e["port"]) as server:
            server.starttls(context=ctx)
            server.login(e["user"], e["app_password"])
            server.send_message(msg)

        return True
    except Exception as ex:
        st.error(f"Email error: {ex}")
        return False


#  alerts (sms)

def send_sms(text):
    try:
        from twilio.rest import Client
        t = st.secrets["twilio"]
        client = Client(t["sid"], t["token"])
        client.messages.create(body=text, from_=t["from_"], to=t["to"])
        return True
    except Exception as ex:
        st.error(f"SMS error: {ex}")
        return False
    
#     [email]
# user = "9b883e001@smtp-brevo.com"
# app_password = "gETJy95b41hQP6ap"
# smtp = "smtp-relay.brevo.com"
# port = 587
# to = "shosheasanin@gmail.com"

# notify new alert 
if "last_alert_ts" not in st.session_state:
    st.session_state.last_alert_ts = None

if not hits.empty:
    ts = str(hits.iloc[-1]["timestamp"])
    if ts != st.session_state.last_alert_ts:
        st.session_state.last_alert_ts = ts
        body = f"Alert! {hits.iloc[-1]['station_id']} | {hits.iloc[-1]['variable']} | value={hits.iloc[-1]['value']}"
        send_sms(body)
        st.success("SMS alert sent!")


# # --- test email button ---
# if st.button("Send test email"):
#     ok = send_email("Test alert", "This is a test from your Streamlit app.")
#     if ok:
#         st.success("Test email sent successfully!")
#     else:
#         st.error("Failed to send test email. Check your SMTP settings.")

if st.button("Send test SMS"):
    if send_sms("Test message from your Digital Twin app!"):
        st.success("Test SMS sent!")
    else:
        st.error("SMS failed")



