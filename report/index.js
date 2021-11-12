//extend dayjs https://day.js.org/docs/en/durations/humanize
dayjs.extend(window.dayjs_plugin_duration);
dayjs.extend(window.dayjs_plugin_relativeTime);

let data;
let graphviz;

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

// return the status in correct color and the duration if we have space
const formatStatus = (data, showDuration) => {
  if (!data.status) {
    return "";
  }
  let color = colors.N9;
  let sep = " - after ";
  switch (data.status) {
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
  let text = data.status;
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
            <td align="left">${dayjs(date).format("M/D/YYYY h:mm:ss A")}</td>
            <td></td>
          </tr>
          `;
};

// return a date range
const formatDateRange = (startDate, endDate) => {
  if (!startDate) {
    return "";
  }

  const dayFormat = "M/D/YYYY";
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
          <td align="left">${label}:${" "}${value}</td>
          <td></td>
        </tr>`;
};

// convert the data to a node
const getTable = (data) => {
  let ret = `label=<
        <table cellspacing="0" cellpadding="2" border="0" color="${
          colors.B10
        }" cellborder="0" bgcolor="${colors.white}">
          <tr>
            <td align="left" bgcolor="${colors.B10}" id="expando;${
    data.unique_id
  }" href=" "><img src="${
    data.open ? "/report/close.svg" : "/report/open.svg"
  }" /></td>
            <td bgcolor="${colors.B10}" ><font point-size="16" color="${
    colors.N2
  }">${data.step_name}</font></td>
        <td bgcolor="${colors.B10}">${"   "}</td>
          </tr>
          <!-- Some extra space at the top -->
          ${data.open ? `<tr><td>${" "}</td></tr>` : null}
          ${data.open ? formatStatus(data, true) : formatStatus(data)}
          ${data.open ? formatKey(data.unique_id) : null}
          ${data.open ? formatText("Type", data.step_class_name) : null}
          ${data.open ? formatText("Version", data.version) : null}
          ${data.open ? formatDateRange(data.start_time, data.end_time) : null}
          ${data.open ? formatLink("Results", data.result_location) : null}
          ${data.open ? formatText("Error", data.error) : null}
          <!-- Some extra space at the bottom -->
          ${data.open ? `<tr><td>${" "}</td></tr>` : null}
        </table>
      >`;
  return ret;
};

// convert the data to dot format
const convert = (json) => {
  let nodes = [];
  let edges = [];
  Object.entries(json).forEach(([k, v]) => {
    nodes.push(`"${k}" [id="${k}" tooltip=" " ${getTable(v)}];`);
    v.dependencies.forEach((d) =>
      edges.push(`"${d}" -> "${k}" [id="${d}->${k}" tooltip=" "];`)
    );
  });
  return `${nodes.join("\n")} ${edges.join("\n")}`;
};

// http://bl.ocks.org/magjac/b2bf6da945e725a605e0d077781457b2
startApp = () => {
  var expandos = d3.selectAll('*[id^="a_expando"]');
  var keys = d3.selectAll('*[id^="a_key"]');

  expandos.on("click", function () {
    var event = d3.event;
    event.preventDefault();
    event.stopPropagation();
    if (event.which == 1) {
      var id = d3.select(this).attr("id").split(";")[1];
      data[id].open = !data[id].open;
      render(data);
    }
  });

  keys.on("click", function () {
    var event = d3.event;
    event.preventDefault();
    event.stopPropagation();
    if (event.which == 1) {
      var id = d3.select(this).attr("id").split(";")[1];
      navigator.clipboard.writeText(id);
      alert(`ID Copied to clipboard: ${id}`);
      // todo: test on server (fails locally)
    }
  });
};

const render = (d) => {
  let digraph = `
        digraph "" {
          bgcolor="transparent"
          node [shape=none fontcolor="${
            colors.N9
          }" fontsize="12" fontname="Helvetica"];
          edge [penwidth=2.0 color="${colors.N9}"]
          concentrate=True;
          rankdir=TB;

          ${convert(d || data)}
        }
        `;

  graphviz
    .zoomScaleExtent([0.2, 1])
    .addImage("/report/open.svg", "32px", "32px")
    .addImage("/report/close.svg", "32px", "32px")
    .renderDot(digraph, startApp);
};

// TODO: we need to reload every few seconds (and keep local open node state)
fetch("/api/steps", {
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
})
  .then((res) => res.json())
  .then((out) => {
    data = out[Object.keys(out)[0]];
    graphviz = d3
      .select("#chart")
      .graphviz()
      .on("initEnd", render)
      .transition(() => {
        return d3.transition().duration(750);
      });
  })
  .catch((err) => {
    throw err;
  });
