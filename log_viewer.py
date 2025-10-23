from flask import Flask, render_template_string, request
import os
from datetime import datetime

app = Flask(__name__)
LOG_FILE = "/var/log/cryptoml.log"

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CryptoML Log Viewer</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .log-text {
      font-family: monospace;
      white-space: pre-wrap;
      background: #0f172a;
      color: #f8fafc;
      border-radius: 0.5rem;
      padding: 1rem;
    }
    .highlight { background-color: #f87171; color: #fff; }
  </style>
</head>
<body class="bg-gray-100 min-h-screen flex flex-col items-center py-10">
  <div class="bg-white shadow-xl rounded-2xl p-6 w-full max-w-5xl">
    <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">📊 CryptoML Log Viewer</h1>

    <form method="get" class="flex flex-wrap items-center justify-center gap-4 mb-8">
      <div>
        <label class="font-medium text-gray-700">📅 Pilih tanggal:</label>
        <input type="date" name="date" value="{{ selected_date }}" class="border rounded-md px-2 py-1">
      </div>
      <div>
        <label class="font-medium text-gray-700">🔍 Cari:</label>
        <input type="text" id="search" placeholder="Ketik misal 'error'..." class="border rounded-md px-2 py-1">
      </div>
      <button type="submit" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg shadow">Tampilkan</button>
    </form>

    {% if sessions %}
      <div id="sessions">
        {% for s in sessions %}
        <details class="mb-4 border rounded-lg shadow-sm bg-gray-50 hover:bg-gray-100">
          <summary class="cursor-pointer px-4 py-2 font-semibold text-gray-800">
            🕒 {{ s['start'] }} — {{ s['end'] }}
          </summary>
          <div class="p-4 log-text text-sm" data-content="true">{{ s['content'] }}</div>
        </details>
        {% endfor %}
      </div>
    {% else %}
      <p class="text-center text-gray-500 italic">Tidak ada log untuk tanggal ini.</p>
    {% endif %}
  </div>

  <script>
    const searchInput = document.getElementById("search");
    const logBlocks = document.querySelectorAll("[data-content]");
    if (searchInput && logBlocks.length > 0) {
      searchInput.addEventListener("input", () => {
        const query = searchInput.value.toLowerCase();
        logBlocks.forEach(block => {
          const lines = block.textContent.split("\\n");
          const filtered = lines.filter(line => line.toLowerCase().includes(query));
          block.innerHTML = filtered.map(line => {
            if (query && line.toLowerCase().includes(query)) {
              return "<span class='highlight'>" + line + "</span>";
            }
            return line;
          }).join("\\n");
        });
      });
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    selected_date = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    sessions = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            content = f.read()

        # Ambil hanya log yang berisi tanggal yang diminta
        if selected_date in content:
            lines = content.splitlines()
        else:
            # fallback ambil semua jika tidak ditemukan tanggal
            lines = content.splitlines()

        current_session = {"start": None, "end": None, "content": ""}
        for line in lines:
            if "Run started" in line:
                # mulai sesi baru
                if current_session["start"]:
                    sessions.append(current_session)
                current_session = {"start": line.strip(), "end": None, "content": ""}
            elif "Run finished" in line:
                current_session["end"] = line.strip()
                sessions.append(current_session)
                current_session = {"start": None, "end": None, "content": ""}
            else:
                if current_session["start"]:
                    current_session["content"] += line + "\n"

    return render_template_string(TEMPLATE, sessions=sessions, selected_date=selected_date)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
