document.addEventListener("DOMContentLoaded", () => {
  // Tab functionality
  const tabBtns = document.querySelectorAll(".tab-btn")
  const tabContents = document.querySelectorAll(".tab-content")

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      // Remove active class from all buttons and contents
      tabBtns.forEach((b) => b.classList.remove("active"))
      tabContents.forEach((c) => c.classList.remove("active"))

      // Add active class to clicked button and corresponding content
      btn.classList.add("active")
      const tabId = btn.getAttribute("data-tab")
      document.getElementById(`${tabId}-tab`).classList.add("active")
    })
  })

  // Elements
  const countrySelect = document.getElementById("country-select")
  const analyzeBtn = document.getElementById("analyze-btn")
  const channelUrlInput = document.getElementById("channel-url")
  const analyzeChannelBtn = document.getElementById("analyze-channel-btn")
  const loadingElement = document.getElementById("loading")
  const resultsElement = document.getElementById("results")
  const channelResultsElement = document.getElementById("channel-results")
  const errorElement = document.getElementById("error-message")

  // Trending analysis elements
  const countryNameElement = document.getElementById("country-name")
  const insightsList = document.getElementById("insights-list")
  const categoryDistImg = document.getElementById("category-dist")
  const viewsByCategoryImg = document.getElementById("views-by-category")
  const publishTimeHeatmapImg = document.getElementById("publish-time-heatmap")
  const durationAnalysisImg = document.getElementById("duration-analysis")
  const tagsAnalysisImg = document.getElementById("tags-analysis")
  const topChannelsImg = document.getElementById("top-channels")

  // Channel analysis elements
  const channelNameElement = document.getElementById("channel-name")
  const channelInsightsList = document.getElementById("channel-insights-list")
  const videoPerformanceImg = document.getElementById("video-performance")
  const channelCategoriesImg = document.getElementById("channel-categories")
  const channelUploadTimesImg = document.getElementById("channel-upload-times")
  const channelDurationImg = document.getElementById("channel-duration")
  const channelEngagementImg = document.getElementById("channel-engagement")
  const improvementTips = document.getElementById("improvement-tips")

  // Function to show error
  function showError(message) {
    errorElement.textContent = message
    errorElement.style.display = "block"
    loadingElement.style.display = "none"

    // Scroll to error message
    errorElement.scrollIntoView({ behavior: "smooth" })
  }

  // Function to reset UI state
  function resetUI() {
    loadingElement.style.display = "flex"
    resultsElement.style.display = "none"
    channelResultsElement.style.display = "none"
    errorElement.style.display = "none"
  }

  // Trending analysis
  analyzeBtn.addEventListener("click", () => {
    const selectedCountry = countrySelect.value
    resetUI()

    // Make API request
    fetch("/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        country: selectedCountry,
      }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok")
        }
        return response.json()
      })
      .then((data) => {
        loadingElement.style.display = "none"

        if (data.error) {
          showError(data.error)
          return
        }

        // Update country name
        countryNameElement.textContent = data.country

        // Populate insights
        insightsList.innerHTML = ""
        data.insights.forEach((insight) => {
          const p = document.createElement("p")
          p.textContent = insight
          insightsList.appendChild(p)
        })

        // Update chart images
        categoryDistImg.src = data.plots.category_dist
        viewsByCategoryImg.src = data.plots.views_by_category
        publishTimeHeatmapImg.src = data.plots.publish_time_heatmap
        durationAnalysisImg.src = data.plots.duration_analysis
        tagsAnalysisImg.src = data.plots.tags_analysis
        topChannelsImg.src = data.plots.top_channels

        // Show results
        resultsElement.style.display = "block"

        // Scroll to results
        resultsElement.scrollIntoView({ behavior: "smooth" })
      })
      .catch((error) => {
        showError("An error occurred while analyzing the data. Please try again.")
        console.error("Error:", error)
      })
  })

  // Channel analysis
  analyzeChannelBtn.addEventListener("click", () => {
    const channelUrl = channelUrlInput.value.trim()

    if (!channelUrl) {
      showError("Please enter a valid YouTube channel URL.")
      return
    }

    resetUI()

    // Make API request for channel analysis
    fetch("/analyze-channel", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        channel_url: channelUrl,
      }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok")
        }
        return response.json()
      })
      .then((data) => {
        loadingElement.style.display = "none"

        if (data.error) {
          showError(data.error)
          return
        }

        // Update channel name
        channelNameElement.textContent = data.channel_name

        // Populate channel insights
        channelInsightsList.innerHTML = ""
        data.insights.forEach((insight) => {
          const p = document.createElement("p")
          p.textContent = insight
          channelInsightsList.appendChild(p)
        })

        // Update channel chart images
        videoPerformanceImg.src = data.plots.video_performance
        channelCategoriesImg.src = data.plots.channel_categories
        channelUploadTimesImg.src = data.plots.upload_times
        channelDurationImg.src = data.plots.duration_analysis
        channelEngagementImg.src = data.plots.engagement_metrics

        // Populate improvement tips with animation
        improvementTips.innerHTML = ""
        data.improvement_tips.forEach((tip, index) => {
          const tipCard = document.createElement("div")
          tipCard.className = "tip-card"
          tipCard.style.opacity = "0"
          tipCard.style.transform = "translateY(20px)"

          const tipTitle = document.createElement("h4")
          tipTitle.textContent = tip.title

          const tipContent = document.createElement("p")
          tipContent.textContent = tip.content

          tipCard.appendChild(tipTitle)
          tipCard.appendChild(tipContent)
          improvementTips.appendChild(tipCard)

          // Animate tips appearing with a delay
          setTimeout(() => {
            tipCard.style.transition = "opacity 0.5s ease, transform 0.5s ease"
            tipCard.style.opacity = "1"
            tipCard.style.transform = "translateY(0)"
          }, 100 * index)
        })

        // Show channel results
        channelResultsElement.style.display = "block"

        // Scroll to channel results
        channelResultsElement.scrollIntoView({ behavior: "smooth" })
      })
      .catch((error) => {
        showError("An error occurred while analyzing the channel. Please try again.")
        console.error("Error:", error)
      })
  })

  // Add enter key support for channel URL input
  channelUrlInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      analyzeChannelBtn.click()
    }
  })
})

