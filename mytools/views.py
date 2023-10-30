from django.shortcuts import render
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = []

# Load data from the CSV file

with open("mytools\\media\\data.csv", encoding='iso-8859-1', mode="r", newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

def index(request):
    result = []
    if request.method == "POST":
        query = request.POST.get("query", "")
        if query:
            result = get_recommendations(query)
    context = {"resul": result}
    return render(request, "index.html", context)

def get_recommendations(tool_name):
    tool_name = tool_name.lower()
    tool_names = [row["Ai Tools Name"].lower() for row in data]
    tool_names.append(tool_name)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(tool_names)
    cosine_similarities = linear_kernel(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    similar_tool_indices = cosine_similarities.argsort()[:-11:-1]
    recommended_tools = []

    for i in similar_tool_indices:
        tool = data[i]
        recommended_tool = [
            tool["Ai Tools Name"],
            tool["Site url"],
            tool["Categories"],
            tool["Per Month"],
            tool["Types For Use"]
        ]
        recommended_tools.append(recommended_tool)

    return recommended_tools
