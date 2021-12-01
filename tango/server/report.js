//extend dayjs https://day.js.org/docs/en/durations/humanize
dayjs.extend(window.dayjs_plugin_duration);
dayjs.extend(window.dayjs_plugin_relativeTime);

// keep track of open nodes across iterations
const state = { data: {}, graphviz: undefined, openMap: {} };

const states = {
  COMPLETED: "completed",
  FAILED: "failed",
  RUNNING: "running",
  INCOMPLETE: "incomplete",
};

// colors
const colors = {
  B2: "#D5EAFE",
  B6: "#265ED4",
  B10: "#223367",
  G6: "#1EC28E",
  G8: "#0A8F6B",
  N2: "#F8F9FA",
  N9: "#47515C",
  O8: "#FF9100",
  R6: "#F7605F",
  R8: "#D63F3F",
  white: "#FFFFFF",
};

// pop a temportal notfication to user
const tempAlert = (msg, duration) => {
  let el = document.createElement("div");
  el.className = "notification";
  el.innerHTML = msg;
  setTimeout(function () {
    el.parentNode.removeChild(el);
  }, duration);
  document.body.appendChild(el);
};

// return the state in correct color and the duration if we have space
const formatState = (data, showDuration) => {
  if (!data.state) {
    return "";
  }
  let color = colors.N9;
  let sep = " - after ";
  switch (data.state) {
    case states.COMPLETED:
      color = colors.G8;
      break;
    case states.FAILED:
      color = colors.R8;
      break;
    case states.RUNNING:
    case states.INCOMPLETE:
      color = colors.O8;
      sep = " - so far ";
      break;
  }
  // display the duration if it makes sense
  let text = data.state;
  if (showDuration && data.start_time) {
    text += sep;
    text += dayjs
      .duration(
        (data.end_time ? dayjs(data.end_time) : dayjs()).diff(data.start_time)
      )
      .humanize(false);
  }
  return `
          <tr>
            <td></td>
            <td align="left"><font color="${color}">${text}</font></td>
            <td></td>
          </tr>
          `;
};

// return a basic date
const formatDate = (label, date) => {
  if (!date) {
    return "";
  }
  return `
        <tr>
            <td></td>
            <td align="left">${dayjs(date).format("YYYY-MM-DD h:mm:ss A")}</td>
            <td></td>
          </tr>
          `;
};

// return a date range
const formatDateRange = (startDate, endDate) => {
  if (!startDate) {
    return "";
  }

  const dayFormat = "YYYY-MM-DD";
  const timeFormat = "h:mm:ss A";
  const fullFormat = `${dayFormat} ${timeFormat}`;
  let endFormat = fullFormat;
  if (dayjs(startDate).format(dayFormat) === dayjs(endDate).format(dayFormat)) {
    // if the same day, jsut show the time difference
    endFormat = timeFormat;
  }

  return `
        <tr>
            <td></td>
            <td align="left">${dayjs(startDate).format(fullFormat)} - ${dayjs(
    endDate
  ).format(endFormat)}</td>
          <td></td>
          </tr>
          `;
};

// return a blue text with a href
const formatLink = (label, url) => {
  if (!url) {
    return "";
  }
  return `
        <tr>
          <td></td>
          <td align="left" href="${url}" target="_blank"><font color="${
    colors.N9
  }">${label}:</font>${" "}<font color="${colors.B6}">${url}</font></td>
          <td></td>
        </tr>`;
};

// return a key (has a special id for copy to clipboard later)
// the `id="key;${value}" href=" "` is a hack to allow selecting ny id later
const formatKey = (value) => {
  if (!value) {
    return "";
  }
  return `
        <tr>
          <td></td>
          <td align="left" id="key;${value}" href=" ">${value}</td>
          <td></td>
        </tr>`;
};

// return basic text
const formatText = (label, value) => {
  if (!value) {
    return "";
  }
  return `
        <tr>
          <td></td>
          <td align="left">${label ? label + ": " : ""}${value}</td>
          <td></td>
        </tr>`;
};

// convert the data to a node
const getTable = (data) => {
  const isOpen = state.openMap[data.unique_id];
  let ret = `label=<
        <table cellspacing="0" cellpadding="2" border="0" color="${
          colors.B10
        }" cellborder="0" bgcolor="${colors.white}">
          <tr>
            <td align="left" bgcolor="${colors.B10}" id="expando;${
    data.unique_id
  }" href=" "><img src="${isOpen ? "/close.svg" : "/open.svg"}" /></td>
            <td bgcolor="${colors.B10}" ><font point-size="16" color="${
    colors.N2
  }">${data.step_name}</font></td>
        <td bgcolor="${colors.B10}">${"   "}</td>
          </tr>
          <!-- Some extra space at the top -->
          ${isOpen ? `<tr><td>${" "}</td></tr>` : null}
          ${isOpen ? formatState(data, true) : formatState(data)}
          ${isOpen ? formatText(undefined, data.unique_id) : null}
          ${isOpen ? formatText("Type", data.step_class_name) : null}
          ${isOpen ? formatText("Version", data.version) : null}
          ${isOpen ? formatDateRange(data.start_time, data.end_time) : null}
          ${
            isOpen && data.state !== states.FAILED
              ? formatLink("Results", data.result_location)
              : null
          }
          ${isOpen ? formatText("Error", data.error) : null}
          <!-- Some extra space at the bottom -->
          ${isOpen ? `<tr><td>${" "}</td></tr>` : null}
        </table>
      >`;
  return ret;
};

// convert the data to dot format
const convert = (json) => {
  let nodes = [];
  let edges = [];
  Object.entries(json).forEach(([k, v]) => {
    nodes.push(`"${k}" [id="${k}" tooltip="${v.step_name}" ${getTable(v)}];`);
    v.dependencies.forEach((d) => {
      let tooltip = `${json[d] ? json[d].step_name : "?"} -> ${v.step_name}`;
      return edges.push(
        `"${d}" -> "${k}" [id="${d}->${k}" tooltip="${tooltip}"];`
      );
    });
  });
  return `${nodes.join("\n")} ${edges.join("\n")}`;
};

// http://bl.ocks.org/magjac/b2bf6da945e725a605e0d077781457b2
startApp = () => {
  const expandos = d3.selectAll('*[id^="a_expando"]');

  expandos.on("click", function () {
    var event = d3.event;
    event.preventDefault();
    event.stopPropagation();
    if (event.which == 1) {
      var id = d3.select(this).attr("id").split(";")[1];
      // keep track of open nodes across iterations
      state.openMap[id] = !state.openMap[id];
      render();
    }
  });

  // allow selection of text nodes for copy paste
  const nodes = document.querySelectorAll(".node");
  function nodeClickHandler(evt) {
    const selection = document.getSelection();
    if (selection.rangeCount > 0) {
      selection.removeAllRanges();
    }
    let range = document.createRange();
    range.selectNode(evt.target);
    selection.addRange(range);
    range = selection.getRangeAt(0);
    range.selectNode(evt.target);
    if (range.toString().trim() !== "") {
      document.execCommand("copy");
      tempAlert(
        `<span class="highlight">${range}</span> copied to clipboard.`,
        3000
      );
    }
  }
  nodes.forEach(function (elem) {
    elem.addEventListener("click", nodeClickHandler);
  });
};

const render = () => {
  let digraph = `
        digraph "" {
          bgcolor="transparent"
          node [shape=none fontcolor="${
            colors.N9
          }" fontsize="12" fontname="Helvetica"];
          edge [penwidth=2.0 color="${colors.N9}"]
          concentrate=True;
          rankdir=TB;

          ${convert(state.data)}
        }
        `;
  // set zoom based on amount of stuff displayed
  const filledPositions =
    Object.values(state.openMap).filter((v) => v === true).length * 4 +
    Object.keys(state.data).length;
  const scale = Math.min(1, 0.075 * filledPositions);
  state.graphviz.scale(scale).renderDot(digraph, startApp);
};

const getData = () => {
  fetch("/api/stepinfo", {
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
  })
    .then((res) => res.json())
    .then((out) => {
      const runId = window.location.pathname.split("/")[2];
      const newData = out[runId];
      // this equality check assumes the data comes back in the same order
      if (JSON.stringify(newData) !== JSON.stringify(state.data)) {
        state.data = out[Object.keys(out)[0]];
        render();
      }
    })
    .catch((err) => {
      throw err;
    });
};

const tans = d3.transition();

state.graphviz = d3
  .select("#chart")
  .graphviz()
  .zoomScaleExtent([0.2, 5])
  .addImage("/open.svg", "32px", "32px")
  .addImage("/close.svg", "32px", "32px")
  .on("initEnd", render)
  .transition(tans);

// initial load
getData();

// reload every x seconds
const secondBetweenRenders = 5;
setInterval(function () {
  getData();
}, secondBetweenRenders * 1000);
