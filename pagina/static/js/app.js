const BASE_URL = "http://ec2-98-80-213-246.compute-1.amazonaws.com:3000/";

$(document).ready(function () {
  function loadInfo() {
    $.ajax({
      url: BASE_URL + "model_info",
      method: "GET",
      success: function (data) {
        console.log("model_info:", data);

        $("#infoTracking").text(data?.tracking_uri ?? "—");
        $("#infoRunId").text(data?.run_id ?? "—");
        console.log(data);
        renderMetrics(data?.metrics);
        renderParamsTable(data?.formatted_params);
      },
      error: function (xhr, status, error) {
        alert("Error cargando datos del modelo: " + error);
      },
    });
  }

  function renderMetrics(metrics) {
    const container = $("#metrics-container");
    container.empty();

    const metricMap = {
      recall_macro: "Recall",
      f1: "F1",
      precision: "Precisión",
      accuracy: "Accuracy",
      auc: "Área bajo la curva (AUC)",
    };

    for (const key in metricMap) {
      const raw = metrics?.[key];
      const value =
        raw === null || raw === undefined || raw === ""
          ? "—"
          : Number(raw).toFixed(4);

      const itemHtml = `
        <div class="metric-item">
          <div class="metric-label">${metricMap[key]}</div>
          <div class="metric-value">${value}</div>
        </div>`;
      container.append(itemHtml);
    }
  }

  function renderParamsTable(params) {
    console.log("formatted_params:", params);
    const table = $("#params-table");
    table.empty();
  
    // Fallback: si viene como objeto {k:v}, conviértelo a [{name, value}]
    if (!Array.isArray(params) && params && typeof params === "object") {
      params = Object.entries(params).map(([k, v]) => ({
        name: k,
        value: String(v),
      }));
    }
  
    if (!Array.isArray(params) || params.length === 0) {
      table.html("<tbody><tr><td>No se encontraron parámetros.</td></tr></tbody>");
      return;
    }
  
    let thead = `
      <thead>
        <tr>
          <th>Parámetro</th>
          <th>Valor</th>
        </tr>
      </thead>`;
  
    let tbody = "<tbody>";
    params.forEach((p) => {
      const name = p?.name ?? "—";
      const value = p?.value ?? "—";
      tbody += `
        <tr>
          <th>${name}</th>
          <td>${value}</td>
        </tr>`;
    });
    tbody += "</tbody>";
  
    table.html(thead + tbody);
  }

  function parseInputsFromTextarea(text) {
    return (text || "")
      .split(/\r?\n/)
      .map(s => s.trim())
      .filter(s => s.length > 0);
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"'`=\/]/g, function (c) {
      return ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;",
        "'": "&#39;", "/": "&#x2F;", "`": "&#x60;", "=": "&#x3D;"
      })[c];
    });
  }

  /**
   * Modificado para mostrar "Sentimiento positivo/negativo" y la confianza del modelo.
   */
  function renderPredictTable($container, payload) {
    const probs = payload.probabilities || null;

    let html = '<table class="results-table"><thead><tr>' +
               '<th>#</th><th>Input</th><th>Predicción</th>';

    if (probs && Array.isArray(probs)) {
      html += '<th>Confianza del Modelo</th>'; // Cambiado el título de la columna
    }
    html += '</tr></thead><tbody>';

    for (let i = 0; i < payload.inputs.length; i++) {
      const inp = payload.inputs[i];
      const pred = payload.predictions?.[i];

      // Mapear la predicción (0 o 1) a un texto descriptivo
      const predictionText = pred === 1 
        ? "Sentimiento positivo" 
        : (pred === 0 ? "Sentimiento negativo" : String(pred));
      
      html += `<tr><td>${i+1}</td><td>${escapeHtml(inp)}</td><td>${predictionText}</td>`;

      if (probs && probs[i] && Array.isArray(probs[i])) {
        // Calcular la confianza como la probabilidad máxima de la predicción
        const confidence = Math.max(...probs[i]);
        const confidenceText = (confidence * 100).toFixed(2) + '%';
        html += `<td>${confidenceText}</td>`;
      }
      html += '</tr>';
    }

    html += '</tbody></table>';
    $container.html(html);
  }

  $("#btnPredict").on("click", function () {
    const inputs = parseInputsFromTextarea($("#txtInputs").val());
    const $status = $("#predictStatus");
    const $out = $("#predictOutput");

    if (inputs.length === 0) {
      alert("Escribe al menos un texto (uno por línea).");
      return;
    }

    const loadingMessages = [
        "Tokenizando texto...",
        "Analizando sintaxis...",
        "Aplicando bigramas...",
        "Analizando semántica...",
        "Generando predicción...",
        "Descifrando caritas...",
    ];
    let messageIndex = 0;
    
    // Inicia el ciclo de mensajes de carga
    $status.text(loadingMessages[messageIndex]);
    const loadingInterval = setInterval(() => {
        messageIndex = (messageIndex + 1) % loadingMessages.length;
        $status.text(loadingMessages[messageIndex]);
    }, 1500); // Cambia el mensaje cada 700ms

    $out.empty();

    $.ajax({
      url: BASE_URL + "predict",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ inputs }),
      success: function (data) {
        clearInterval(loadingInterval); // Detiene los mensajes de carga
        renderPredictTable($out, data);
        $status.text(`OK • Run: ${data.model_run_id}`);
      },
      error: function (xhr, status, error) {
        clearInterval(loadingInterval); // Detiene los mensajes de carga en caso de error
        console.error("Error en /predict:", error);
        $status.text("Error");
        $out.html(`<pre class="api-output error">Error: ${xhr.responseText || error}</pre>`);
      }
    });
  });

  const __charts = {};

function createBarChart(canvasId, series, metricLabel, titleText) {
  const el = document.getElementById(canvasId);
  if (!el) return;

  // Si ya existía un chart en este canvas, destrúyelo antes de crear el nuevo
  if (__charts[canvasId]) {
    __charts[canvasId].destroy();
  }

  // 1) Preparamos labels (multilínea) y datos
  //    - Rompe en varias líneas cada vez que encuentre " + " para que se lean completo
  const labelsWrapped = (series.labels || []).map(l =>
    String(l).replace(/ \+ /g, "\n")
  );
  const data = series.data || [];

  // 2) Decidir orientación: horizontal si hay muchos labels
  const horizontal = labelsWrapped.length > 12;

  // 3) Ajustar altura del contenedor dinámicamente si es horizontal
  //    ~24px por barra + margen superior/inferior
  if (horizontal) {
    const container = el.closest(".chart-container");
    if (container) {
      const base = 80; // título + padding
      const perBar = 24; // alto por barra
      const h = Math.max(320, base + perBar * labelsWrapped.length);
      container.style.height = `${h}px`;
      // También permitimos scroll por si igual resulta muy alto en móviles
      container.style.overflow = "auto";
    }
  }

  // 4) Construir gráfico
  __charts[canvasId] = new Chart(el, {
    type: "bar",
    data: {
      labels: labelsWrapped,
      datasets: [{
        label: metricLabel || "Score",
        data: data,
        // Deja que Chart.js elija los colores por defecto o tu tema global
        borderWidth: 1
      }]
    },
    options: {
      indexAxis: horizontal ? "y" : "x",
      responsive: true,
      maintainAspectRatio: false,
      layout: {
        padding: { top: 8, right: 8, bottom: 8, left: 8 }
      },
      plugins: {
        legend: {
          labels: { color: "#F9FAFB" }
        },
        title: {
          display: !!titleText,
          text: titleText || "",
          color: "#F9FAFB",
          font: { size: 14, weight: "600" }
        },
        tooltip: {
          bodyColor: "#F9FAFB",
          titleColor: "#F9FAFB",
          backgroundColor: "rgba(31,41,55,0.95)", // acorde a tema oscuro
          borderColor: "#374151",
          borderWidth: 1
        }
      },
      scales: {
        x: {
          ticks: {
            color: "#9CA3AF",
            autoSkip: false,        // <-- clave: no omitir etiquetas
            maxRotation: 0,
            minRotation: 0
          },
          grid: { color: "#374151" }
        },
        y: {
          ticks: {
            color: "#9CA3AF",
            autoSkip: false         // <-- también en eje Y si horizontal
          },
          grid: { color: "#374151" }
        }
      }
    }
  });
}

function loadAblationChart() {
  $.ajax({
    url: BASE_URL + "ablation_summary",
    method: "GET",
    success: function (response) {
      const uni = response.baseline_unigram;
      const bi  = response.baseline_bigram;

      if (uni && Array.isArray(uni.labels) && uni.labels.length > 0) {
        createBarChart(
          'ablationChartUnigram',
          response.baseline_unigram,
          'recall_macro'
        );
      }

      if (bi && Array.isArray(bi.labels) && bi.labels.length > 0) {
        createBarChart(
          'ablationChartBigram',
          response.baseline_bigram,
          'recall_macro'
        );
      }
    },
    error: function (xhr, status, error) {
      console.error("Error cargando datos de ablación:", error);
    }
  });
}

  function loadComparison() {
    $.ajax({
      url: BASE_URL + "comparison",
      method: "GET",
      success: function (resp) {
        $("#comparisonSummary").text(resp.text_top || "—");

        const headers = resp.table?.headers || [];
        const rows = resp.table?.rows || [];
        if (!headers.length || !rows.length) {
          $("#comparisonTable").html('<tr><td>No hay datos de comparación</td></tr>');
        } else {
          let thead = "<thead><tr>";
          headers.forEach(h => { thead += `<th>${$("<div>").text(String(h)).html()}</th>`; });
          thead += "</tr></thead>";

          let tbody = "<tbody>";
          rows.forEach(r => {
            tbody += "<tr>";
            r.forEach(c => {
              tbody += `<td>${$("<div>").text(String(c)).html()}</td>`;
            });
            tbody += "</tr>";
          });
          tbody += "</tbody>";

          $("#comparisonTable").html(thead + tbody);
        }

        const $bottom = $("#comparisonTextBottom");
        if ($bottom.length) {
          $bottom.text(resp.text_bottom || "—");
        }
      },
      error: function (xhr, status, error) {
        $("#comparisonSummary").text("No se pudo cargar la comparación.");
        $("#comparisonTable").html('<tr><td>Error al cargar la comparación.</td></tr>');
        const $bottom = $("#comparisonTextBottom");
        if ($bottom.length) $bottom.text("");
        console.error("Error en /comparison:", error);
      }
    });
  }
  
  const OTHER_MODELS = [
    {
      key: "logreg_modelofinal",
      label: "Regresión Logistica",
      description:
        "Modelo de Regresión Logística optimizado con solver 'saga' sobre features TF-IDF de bigramas.",
      run_id: "19a0e0342f0a4bdaaab2ebf89b62ca92",
      experiment_id: "568376531369023149",
      raw_metrics: {
        accuracy: 0.827215625,
        precision_weighted: 0.8273442001883925,
        f1_weighted: 0.8271986566489014,
        precision_macro: 0.8273442001883924,
        recall_macro: 0.827215625,
        recall_weighted: 0.827215625,
        f1_macro: 0.8271986566489011
      },
      get metrics() {
        return {
          recall_macro: this.raw_metrics.recall_macro ?? null,
          f1: this.raw_metrics.f1_macro ?? null,
          precision: this.raw_metrics.precision_macro ?? null,
          accuracy: this.raw_metrics.accuracy ?? null,
          auc: null
        };
      },
      params: {
        tfidf_lowercase: false,
        test_size: 0.2,
        model_solver: "saga",
        model_max_iter: 1000,
        tamaño_train: 1280000,
        model_C: 1.0,
        tamaño_test: 320000,
        model: "LogisticRegression",
        model_penalty: "l2",
        random_state: 666,
        tfidf_ngram_range: "(1, 2)",
        vectorizer: "TfidfVectorizer",
        model_n_jobs: -1,
        tfidf_token_pattern: "\\S+",
        tfidf_max_features: "None"
      },
      param_map: {
        tfidf_lowercase: "TF-IDF lowercase",
        test_size: "Test size",
        model_solver: "Solver",
        model_max_iter: "Max iter",
        tamaño_train: "Tamaño Train",
        model_C: "C",
        tamaño_test: "Tamaño Test",
        model: "Modelo",
        model_penalty: "Penalización",
        random_state: "Semilla",
        tfidf_ngram_range: "TF-IDF ngram_range",
        vectorizer: "Vectorizador",
        model_n_jobs: "n_jobs",
        tfidf_token_pattern: "TF-IDF token_pattern",
        tfidf_max_features: "TF-IDF max_features"
      }
    },
    {
      key: "keras_mlp_tfidf_svd",
      label: "Keras-MLP Red Neuronal",
      description:
        "Keras MLP sobre TF-IDF bigrama + TruncatedSVD. Evaluación en TEST: métricas macro/weighted y matrices de confusión.",
      run_id: "ea1dbfe391ee4b7f9f41857daf6ce009",
      experiment_id: "568376531369023149",
      raw_metrics: {
        accuracy: 0.748990625,
        precision_weighted: 0.7490482124278882,
        f1_weighted: 0.7489761139342549,
        precision_macro: 0.7490482124278881,
        recall_macro: 0.748990625,
        recall_weighted: 0.748990625,
        f1_macro: 0.7489761139342548,
        threshold: 0.5
      },
      get metrics() {
        return {
          recall_macro: this.raw_metrics.recall_macro ?? null,
          f1: this.raw_metrics.f1_macro ?? null,
          precision: this.raw_metrics.precision_macro ?? null,
          accuracy: this.raw_metrics.accuracy ?? null,
          auc: null
        };
      },
      params: {
        tfidf__preprocessor: "None",
        input_dim: 500,
        tfidf__strip_accents: "None",
        svd__n_iter: 3,
        tfidf__input: "content",
        tfidf__encoding: "utf-8",
        svd__algorithm: "randomized",
        svd__power_iteration_normalizer: "auto",
        tfidf__binary: false,
        tfidf__stop_words: "None",
        tfidf__smooth_idf: true,
        svd__tol: 0.0,
        tfidf__min_df: 5,
        tfidf__tokenizer: "None",
        tfidf__max_features: "None",
        tfidf__use_idf: true,
        tfidf__token_pattern: "\\S+",
        tfidf__norm: "l2",
        tfidf__dtype: "<class 'numpy.float32'>",
        svd__n_oversamples: 10,
        tfidf__analyzer: "word",
        tfidf__vocabulary: "None",
        optimizer: "AdamW",
        tfidf__ngram_range: "(2, 2)",
        learning_rate: 1.1718750556610757e-06,
        tfidf__decode_error: "strict",
        svd__n_components: 500,
        tfidf__sublinear_tf: false,
        svd__random_state: 666,
        tfidf__lowercase: false,
        tfidf__max_df: 0.9,
        model_total_params: 146817
      },
      param_map: {
        // TF-IDF
        "tfidf__lowercase": "TF-IDF lowercase",
        "tfidf__token_pattern": "TF-IDF token_pattern",
        "tfidf__ngram_range": "TF-IDF ngram_range",
        "tfidf__max_features": "TF-IDF max_features",
        "tfidf__min_df": "TF-IDF min_df",
        "tfidf__max_df": "TF-IDF max_df",
        "tfidf__use_idf": "TF-IDF use_idf",
        "tfidf__smooth_idf": "TF-IDF smooth_idf",
        "tfidf__norm": "TF-IDF norm",
        "tfidf__dtype": "TF-IDF dtype",
        "tfidf__analyzer": "TF-IDF analyzer",
        "tfidf__binary": "TF-IDF binary",
        "tfidf__stop_words": "TF-IDF stop_words",
        "tfidf__preprocessor": "TF-IDF preprocessor",
        "tfidf__tokenizer": "TF-IDF tokenizer",
        "tfidf__input": "TF-IDF input",
        "tfidf__encoding": "TF-IDF encoding",
        "tfidf__decode_error": "TF-IDF decode_error",
        "tfidf__vocabulary": "TF-IDF vocabulary",
        "tfidf__sublinear_tf": "TF-IDF sublinear_tf",
        // SVD
        "svd__n_components": "SVD n_components",
        "svd__n_iter": "SVD n_iter",
        "svd__algorithm": "SVD algorithm",
        "svd__tol": "SVD tol",
        "svd__n_oversamples": "SVD n_oversamples",
        "svd__random_state": "SVD random_state",
        "svd__power_iteration_normalizer": "SVD power_iter_normalizer",
        // MLP
        "input_dim": "Dimensión de entrada (SVD)",
        "optimizer": "Optimizer",
        "learning_rate": "Learning rate",
        "model_total_params": "Parámetros totales del MLP"
      }
    }
  ];

  function mapParamsToTableRows(paramsObj, nameMap) {
    const rows = [];
    for (const [k, v] of Object.entries(paramsObj || {})) {
      const nice = nameMap[k] || k;
      rows.push({ name: nice, value: String(v) });
    }
    return rows.sort((a, b) => a.name.localeCompare(b.name, "es"));
  }

  function renderMetricsInto($container, metricsObj) {
    $container.empty();
    const metricMap = {
      recall_macro: "Recall",
      f1: "F1",
      precision: "Precisión",
      accuracy: "Accuracy",
    };

    for (const key in metricMap) {
      const raw = metricsObj?.[key];
      const val =
        raw === null || raw === undefined || raw === "" ? "—" : Number(raw).toFixed(4);
      const html = `
        <div class="metric-item">
          <div class="metric-label">${metricMap[key]}</div>
          <div class="metric-value">${val}</div>
        </div>`;
      $container.append(html);
    }
  }

  function renderOtherModels() {
    const $root = $("#other-models");
    if (!$root.length) return;

    OTHER_MODELS.forEach((m) => {
      const metricsId = `metrics-${m.key}`;
      const tableId   = `table-${m.key}`;

      const card = $(`
        <div class="card">
          <h2 class="card-title" style="display:flex;align-items:center;gap:10px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
                 viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <path d="M16 8a6 6 0 1 1-8 8"></path>
            </svg>
            ${m.label}
            <span style="font-weight:400;color:#9CA3AF;margin-left:8px;">
              (Exp: ${m.experiment_id} • Run: ${m.run_id})
            </span>
          </h2>

          <p class="card-subtitle" style="margin-top:-6px;">${m.description}</p>

          <div class="card">
            <h3 class="card-title">Métricas clave</h3>
            <div id="${metricsId}" class="metrics-grid"></div>
          </div>

          <div class="card">
            <h3 class="card-title">Parámetros</h3>
            <div class="table-container">
              <table id="${tableId}">
                <thead>
                  <tr><th>Parámetro</th><th>Valor</th></tr>
                </thead>
                <tbody></tbody>
              </table>
            </div>
          </div>
        </div>
      `);

      $root.append(card);

      // Métricas
      renderMetricsInto($(`#${metricsId}`), m.metrics);

      // Parámetros (usa el mapa específico del modelo si existe)
      const rows = mapParamsToTableRows(m.params, m.param_map || {});
      const $tbl = $(`#${tableId}`);
      let tbody = "<tbody>";
      rows.forEach(r => { tbody += `<tr><th>${r.name}</th><td>${r.value}</td></tr>`; });
      tbody += "</tbody>";
      $tbl.find("tbody").replaceWith(tbody);
    });
  }

  // Lógica para la tarjeta desplegable
  //$(".card-title-toggle").on("click", function() {
  //  $(this).toggleClass("active");
  //  $(this).next(".collapsible-content").slideToggle(300);
  //});

  $(document).on("click", ".card-title-toggle", function () {
    $(this).toggleClass("active");
    $(this).next(".collapsible-content").slideToggle(300);
  });
  
  // Llamadas iniciales
  loadInfo();
  loadAblationChart();
  loadComparison();
  renderOtherModels();
});

