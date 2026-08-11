/**
 * Analytics & Physiotherapist Daily Report Handler
 */

document.addEventListener("DOMContentLoaded", () => {
  renderAdherenceChart();
});

function renderAdherenceChart() {
  const canvas = document.getElementById("trendChart");
  if (!canvas || typeof historyData === "undefined" || !historyData) return;

  const labels = historyData.map(item => {
    const d = new Date(item.log_date);
    return d.toLocaleDateString('en-US', { weekday: 'short', month: 'numeric', day: 'numeric' });
  });

  const repsData = historyData.map(item => item.total_reps);
  const accuracyData = historyData.map(item => item.overall_accuracy);

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: labels.length > 0 ? labels : ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
      datasets: [
        {
          label: 'Completed Repetitions',
          data: repsData.length > 0 ? repsData : [25, 30, 42, 38, 50, 45, 60],
          backgroundColor: '#DA7756',
          borderRadius: 8,
          yAxisID: 'y'
        },
        {
          label: 'Form Accuracy %',
          data: accuracyData.length > 0 ? accuracyData : [88, 92, 95, 91, 96, 94, 98],
          type: 'line',
          borderColor: '#10B981',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          fill: true,
          tension: 0.35,
          borderWidth: 3,
          pointRadius: 5,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            font: { family: 'Plus Jakarta Sans', weight: '700', size: 12 },
            usePointStyle: true
          }
        }
      },
      scales: {
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          title: { display: true, text: 'Reps Completed' },
          grid: { color: '#E6E4DD' }
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          min: 50,
          max: 100,
          title: { display: true, text: 'Accuracy %' },
          grid: { drawOnChartArea: false }
        },
        x: {
          grid: { display: false }
        }
      }
    }
  });
}

async function sendPhysioReport(event) {
  event.preventDefault();
  
  const physioEmail = document.getElementById("physio_email").value;
  const patientName = document.getElementById("patient_name").value;
  const therapistNotes = document.getElementById("therapist_notes").value;

  try {
    const response = await fetch('/api/send_report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        physio_email: physioEmail,
        patient_name: patientName,
        therapist_notes: therapistNotes
      })
    });

    const res = await response.json();
    if (res.status === 'success') {
      showToast(`Report successfully sent to ${physioEmail}!`);
    } else {
      alert("Error sending report: " + res.message);
    }
  } catch (err) {
    console.error("Report submission failed:", err);
    alert("Network error. Please try again.");
  }
}

function showToast(message) {
  const toast = document.getElementById("toast");
  const msgEl = document.getElementById("toast-msg");
  if (toast && msgEl) {
    msgEl.textContent = message;
    toast.classList.add("show");
    setTimeout(() => {
      toast.classList.remove("show");
    }, 3500);
  }
}
