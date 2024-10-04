
redis-server --daemonize yes
if [ $? -ne 0 ]; then
    echo "Failed to start Redis server."
    exit 1
fi

streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0
if [ $? -ne 0 ]; then
    echo "Failed to start Streamlit app."
    exit 1
fi