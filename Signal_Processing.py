from scipy.signal import butter, filtfilt

# Function to apply band-pass filtering
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=256, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

st.subheader("Signal Processing and Noise Reduction")
if st.button('Apply Band-Pass Filter'):
    filtered_data = bandpass_filter(X[sample_index])
    st.write("Filtered Data (First 10 Points):", filtered_data[:10])
    
    fig, ax = plt.subplots(facecolor='#0D1117')
    ax.set_facecolor('#0D1117')
    ax.plot(filtered_data, color='cyan')
    ax.set_title(f"Filtered EEG Recording: {sample_index}", fontdict=font_properties)
    ax.set_xlabel("Datapoint (0-1024)", fontdict=font_properties)
    ax.set_ylabel("Voltage", fontdict=font_properties)
    ax.tick_params(colors='white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    st.pyplot(fig)
