from flask import Flask, render_template, request, jsonify
import isodate
import pandas as pd
import numpy as np
import json
import os
import re
from googleapiclient.discovery import build
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)

# Cache for storing data to avoid repeated API calls
cache = {
    'videos': {},
    'category_mapping': {},
    'channel_data': {}
}

# List of available countries with their codes
COUNTRIES = {
    'United States': 'US',
    'United Kingdom': 'GB',
    'Canada': 'CA',
    'Australia': 'AU',
    'India': 'IN',
    'Japan': 'JP',
    'South Korea': 'KR',
    'Brazil': 'BR',
    'France': 'FR',
    'Germany': 'DE',
    'Russia': 'RU',
    'Mexico': 'MX',
    'Spain': 'ES',
    'Italy': 'IT',
    'Netherlands': 'NL'
}

def get_api_key():
    # In production, use environment variables
    return os.environ.get('YOUTUBE_API_KEY', 'your_api_key')

def get_trending_videos(api_key, region_code='US', max_results=50):
    # Check cache first
    cache_key = f"{region_code}_{max_results}"
    if cache_key in cache['videos']:
        return cache['videos'][cache_key]
    
    # Build the YouTube service
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Initialize the list to hold video details
    videos = []

    # Fetch the most popular videos
    request = youtube.videos().list(
        part='snippet,contentDetails,statistics',
        chart='mostPopular',
        regionCode=region_code,
        maxResults=50
    )

    # Paginate through the results if max_results > 50
    while request and len(videos) < max_results:
        response = request.execute()
        for item in response['items']:
            video_details = {
                'video_id': item['id'],
                'title': item['snippet']['title'],
                'description': item['snippet'].get('description', 'No description'),
                'published_at': item['snippet']['publishedAt'],
                'channel_id': item['snippet']['channelId'],
                'channel_title': item['snippet']['channelTitle'],
                'category_id': item['snippet']['categoryId'],
                'tags': item['snippet'].get('tags', []),
                'duration': item['contentDetails']['duration'],
                'definition': item['contentDetails']['definition'],
                'caption': item['contentDetails'].get('caption', 'false'),
                'view_count': int(item['statistics'].get('viewCount', 0)),
                'like_count': int(item['statistics'].get('likeCount', 0)),
                'comment_count': int(item['statistics'].get('commentCount', 0))
            }
            videos.append(video_details)

        # Get the next page token
        request = youtube.videos().list_next(request, response)

    # Cache the results
    cache['videos'][cache_key] = videos[:max_results]
    return videos[:max_results]

def get_category_mapping(api_key, region_code='US'):
    # Check cache first
    cache_key = f"categories_{region_code}"
    if cache_key in cache['category_mapping']:
        return cache['category_mapping'][cache_key]
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.videoCategories().list(
        part='snippet',
        regionCode=region_code
    )
    response = request.execute()
    category_mapping = {}
    for item in response['items']:
        category_id = item['id']
        category_name = item['snippet']['title']
        category_mapping[category_id] = category_name
    
    # Cache the results
    cache['category_mapping'][cache_key] = category_mapping
    return category_mapping

def extract_channel_id(channel_url):
    """Extract channel ID from various YouTube channel URL formats"""
    youtube_url_patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/channel\/([a-zA-Z0-9_-]+)',  # /channel/UCxxx
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/user\/([a-zA-Z0-9_-]+)',      # /user/username
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/@([a-zA-Z0-9_-]+)'            # /@username
    ]
    
    for pattern in youtube_url_patterns:
        match = re.search(pattern, channel_url)
        if match:
            return match.group(1)
    
    return None

def get_channel_info(api_key, channel_identifier):
    """Get channel info from channel ID or username"""
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    # Check if it's a channel ID (starts with UC) or username
    if channel_identifier.startswith('UC'):
        request = youtube.channels().list(
            part='snippet,statistics,contentDetails',
            id=channel_identifier
        )
    else:
        # Try as a custom URL or username
        request = youtube.channels().list(
            part='snippet,statistics,contentDetails',
            forUsername=channel_identifier
        )
        
        # If no results, try searching for the channel
        response = request.execute()
        if not response.get('items'):
            search_request = youtube.search().list(
                part='snippet',
                q=channel_identifier,
                type='channel',
                maxResults=1
            )
            search_response = search_request.execute()
            if search_response.get('items'):
                channel_id = search_response['items'][0]['id']['channelId']
                request = youtube.channels().list(
                    part='snippet,statistics,contentDetails',
                    id=channel_id
                )
    
    response = request.execute()
    
    if not response.get('items'):
        return None
    
    channel_info = response['items'][0]
    
    return {
        'channel_id': channel_info['id'],
        'title': channel_info['snippet']['title'],
        'description': channel_info['snippet'].get('description', ''),
        'published_at': channel_info['snippet']['publishedAt'],
        'thumbnail': channel_info['snippet']['thumbnails']['default']['url'],
        'view_count': int(channel_info['statistics'].get('viewCount', 0)),
        'subscriber_count': int(channel_info['statistics'].get('subscriberCount', 0)),
        'video_count': int(channel_info['statistics'].get('videoCount', 0)),
        'uploads_playlist': channel_info['contentDetails']['relatedPlaylists']['uploads']
    }

def get_channel_videos(api_key, uploads_playlist_id, max_results=50):
    """Get videos from a channel's uploads playlist"""
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    # Get video IDs from playlist
    video_ids = []
    request = youtube.playlistItems().list(
        part='contentDetails',
        playlistId=uploads_playlist_id,
        maxResults=50
    )
    
    # Paginate through results
    while request and len(video_ids) < max_results:
        response = request.execute()
        
        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])
            
        request = youtube.playlistItems().list_next(request, response)
        
        if len(video_ids) >= max_results:
            break
    
    # Get video details
    videos = []
    # Process in batches of 50 (API limit)
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        request = youtube.videos().list(
            part='snippet,contentDetails,statistics',
            id=','.join(batch)
        )
        response = request.execute()
        
        for item in response['items']:
            try:
                video_details = {
                    'video_id': item['id'],
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', 'No description'),
                    'published_at': item['snippet']['publishedAt'],
                    'channel_id': item['snippet']['channelId'],
                    'channel_title': item['snippet']['channelTitle'],
                    'category_id': item['snippet']['categoryId'],
                    'tags': item['snippet'].get('tags', []),
                    'duration': item['contentDetails']['duration'],
                    'definition': item['contentDetails']['definition'],
                    'caption': item['contentDetails'].get('caption', 'false'),
                    'view_count': int(item['statistics'].get('viewCount', 0)),
                    'like_count': int(item['statistics'].get('likeCount', 0)),
                    'comment_count': int(item['statistics'].get('commentCount', 0))
                }
                videos.append(video_details)
            except KeyError:
                # Skip videos with missing data
                continue
    
    return videos

def process_data(videos, category_mapping):
    df = pd.DataFrame(videos)
    
    # Convert published_at to datetime
    df['published_at'] = pd.to_datetime(df['published_at'])
    
    # Convert ISO 8601 duration to seconds
    df['duration_seconds'] = df['duration'].apply(lambda x: isodate.parse_duration(x).total_seconds())
    
    # Create duration ranges
    df['duration_range'] = pd.cut(
        df['duration_seconds'], 
        bins=[0, 300, 600, 1200, 3600, float('inf')], 
        labels=['0-5 min', '5-10 min', '10-20 min', '20-60 min', '60+ min']
    )
    
    # Map category IDs to names
    df['category_name'] = df['category_id'].map(category_mapping)
    
    # Extract time-related features
    df['publish_hour'] = df['published_at'].dt.hour
    df['publish_day'] = df['published_at'].dt.day_name()
    df['publish_month'] = df['published_at'].dt.month_name()
    
    # Calculate engagement metrics
    df['engagement_score'] = (df['view_count'] + df['like_count']*5 + df['comment_count']*10) / 1000
    df['views_per_day'] = df.apply(
        lambda x: x['view_count'] / max(1, (datetime.now().replace(tzinfo=None) - x['published_at'].to_pydatetime().replace(tzinfo=None)).days), 
        axis=1
    )
    
    # Calculate tag count
    df['tag_count'] = df['tags'].apply(len)
    
    return df

def create_plot(plt_func, **kwargs):
    """Helper function to create and encode plots"""
    plt.figure(figsize=(10, 6))
    plt_func(**kwargs)
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plt.close()
    
    # Encode the image to base64 string
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def generate_insights(df):
    insights = []
    
    # Best time to upload
    best_hour = df.groupby('publish_hour')['view_count'].mean().idxmax()
    insights.append(f"Best hour to upload: {best_hour}:00 (24-hour format)")
    
    best_day = df.groupby('publish_day')['view_count'].mean().idxmax()
    insights.append(f"Best day to upload: {best_day}")
    
    # Best performing categories
    top_categories = df.groupby('category_name')['view_count'].mean().nlargest(3).index.tolist()
    insights.append(f"Top performing categories: {', '.join(top_categories)}")
    
    # Optimal video duration
    best_duration = df.groupby('duration_range')['engagement_score'].mean().idxmax()
    insights.append(f"Optimal video duration: {best_duration}")
    
    # Tag analysis
    optimal_tag_count = df.groupby(pd.cut(df['tag_count'], bins=[0, 5, 10, 15, 20, 100]))['view_count'].mean().idxmax()
    insights.append(f"Optimal tag count: {optimal_tag_count}")
    
    # Top channels
    top_channels = df.groupby('channel_title')['view_count'].mean().nlargest(5).index.tolist()
    insights.append(f"Top trending channels: {', '.join(top_channels[:3])}")
    
    return insights

def generate_channel_insights(df, channel_info):
    insights = []
    
    # Channel overview
    insights.append(f"Total videos: {channel_info['video_count']}")
    insights.append(f"Total subscribers: {channel_info['subscriber_count']:,}")
    insights.append(f"Total views: {channel_info['view_count']:,}")
    
    # Average views per video
    avg_views = df['view_count'].mean()
    insights.append(f"Average views per video: {avg_views:,.0f}")
    
    # Most popular video
    most_popular = df.loc[df['view_count'].idxmax()]
    insights.append(f"Most popular video: '{most_popular['title']}' with {most_popular['view_count']:,} views")
    
    # Upload frequency
    if len(df) > 1:
        date_diffs = df['published_at'].sort_values().diff().dropna()
        avg_days_between = date_diffs.mean().days
        insights.append(f"Upload frequency: ~{avg_days_between:.1f} days between videos")
    
    # Best performing content
    if 'category_name' in df.columns and not df['category_name'].isna().all():
        best_category = df.groupby('category_name')['view_count'].mean().idxmax()
        insights.append(f"Best performing content category: {best_category}")
    
    # Engagement rate
    avg_engagement = df['engagement_score'].mean()
    insights.append(f"Average engagement score: {avg_engagement:.1f}")
    
    return insights

def generate_improvement_tips(df, trending_df=None):
    """Generate actionable tips for channel improvement"""
    tips = []
    
    # Upload timing tips
    if 'publish_hour' in df.columns and 'publish_day' in df.columns:
        best_hour = df.groupby('publish_hour')['view_count'].mean().idxmax()
        best_day = df.groupby('publish_day')['view_count'].mean().idxmax()
        
        # Compare with trending if available
        if trending_df is not None:
            trending_best_hour = trending_df.groupby('publish_hour')['view_count'].mean().idxmax()
            trending_best_day = trending_df.groupby('publish_day')['view_count'].mean().idxmax()
            
            if best_hour != trending_best_hour or best_day != trending_best_day:
                tips.append({
                    "title": "Optimize Upload Schedule",
                    "content": f"Your best performing uploads are on {best_day} at {best_hour}:00, but trending videos perform best on {trending_best_day} at {trending_best_hour}:00. Consider testing this trending schedule."
                })
        else:
            tips.append({
                "title": "Consistent Upload Schedule",
                "content": f"Your videos perform best when uploaded on {best_day} at {best_hour}:00. Try to maintain a consistent schedule around this time."
            })
    
    # Video duration tips
    if 'duration_range' in df.columns:
        best_duration = df.groupby('duration_range')['engagement_score'].mean().idxmax()
        worst_duration = df.groupby('duration_range')['engagement_score'].mean().idxmin()
        
        if best_duration != worst_duration:
            tips.append({
                "title": "Optimize Video Length",
                "content": f"Your {best_duration} videos perform significantly better than your {worst_duration} videos. Consider focusing on the {best_duration} format."
            })
    
    # Content category tips
    if 'category_name' in df.columns and not df['category_name'].isna().all():
        if len(df['category_name'].unique()) > 1:
            best_category = df.groupby('category_name')['view_count'].mean().nlargest(1).index[0]
            tips.append({
                "title": "Content Focus",
                "content": f"Your '{best_category}' videos perform best. Consider creating more content in this category while maintaining your channel's identity."
            })
    
    # Thumbnail and title tips
    if len(df) >= 5:
        # Compare top 20% vs bottom 20% videos
        top_videos = df.nlargest(int(len(df) * 0.2), 'view_count')
        bottom_videos = df.nsmallest(int(len(df) * 0.2), 'view_count')
        
        # Title length analysis
        top_title_length = top_videos['title'].str.len().mean()
        bottom_title_length = bottom_videos['title'].str.len().mean()
        
        if abs(top_title_length - bottom_title_length) > 10:
            better_length = "shorter" if top_title_length < bottom_title_length else "longer"
            tips.append({
                "title": "Title Optimization",
                "content": f"Your top-performing videos have {better_length} titles (avg {top_title_length:.0f} vs {bottom_title_length:.0f} characters). Consider adjusting your title length strategy."
            })
    
    # Tags optimization
    if 'tag_count' in df.columns:
        optimal_tag_count = df.groupby(pd.cut(df['tag_count'], bins=[0, 5, 10, 15, 20, 100]))['view_count'].mean().idxmax()
        avg_tags = df['tag_count'].mean()
        
        if avg_tags < 5:
            tips.append({
                "title": "Improve Video Tags",
                "content": f"You're using only {avg_tags:.1f} tags on average. Using {optimal_tag_count} relevant tags can improve discoverability."
            })
    
    # Upload consistency tips
    if len(df) > 5:
        date_diffs = df['published_at'].sort_values().diff().dropna()
        avg_days_between = date_diffs.mean().days
        max_gap = date_diffs.max().days
        
        if max_gap > avg_days_between * 2:
            tips.append({
                "title": "Consistent Upload Schedule",
                "content": f"Your upload schedule has gaps of up to {max_gap} days. Aim for more consistent uploads around every {avg_days_between:.0f} days to maintain audience engagement."
            })
    
    # Engagement tips
    if 'comment_count' in df.columns and 'view_count' in df.columns:
        avg_comment_ratio = (df['comment_count'] / df['view_count']).mean() * 100
        
        if avg_comment_ratio < 0.5:
            tips.append({
                "title": "Boost Audience Engagement",
                "content": "Your videos have a low comment-to-view ratio. Try asking questions in your videos or adding calls-to-action to encourage more comments and engagement."
            })
    
    # Add general tips if we don't have enough specific ones
    if len(tips) < 3:
        tips.append({
            "title": "Optimize Thumbnails",
            "content": "Use high-contrast, clear thumbnails with minimal text. A/B test different thumbnail styles to see what resonates with your audience."
        })
        
        tips.append({
            "title": "Improve Video Retention",
            "content": "Check your audience retention graphs in YouTube Studio. Identify where viewers drop off and adjust your content pacing accordingly."
        })
    
    return tips

@app.route('/')
def index():
    return render_template('index.html', countries=COUNTRIES)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    country_code = data.get('country', 'US')
    
    try:
        api_key = get_api_key()
        videos = get_trending_videos(api_key, country_code)
        category_mapping = get_category_mapping(api_key, country_code)
        
        if not videos:
            return jsonify({'error': 'No videos found for the selected country'})
        
        df = process_data(videos, category_mapping)
        
        # Generate insights
        insights = generate_insights(df)
        
        # Create visualizations
        plots = {}
        
        # 1. Category Distribution
        plots['category_dist'] = create_plot(
            lambda **kwargs: sns.countplot(y=kwargs['data']['category_name'], 
                                          order=kwargs['data']['category_name'].value_counts().index[:10],
                                          palette='viridis'),
            data=df
        )
        
        # 2. View Count by Category
        plots['views_by_category'] = create_plot(
            lambda **kwargs: sns.barplot(
                y=kwargs['data'].groupby('category_name')['view_count'].mean().nlargest(10).index,
                x=kwargs['data'].groupby('category_name')['view_count'].mean().nlargest(10).values,
                palette='magma'
            ),
            data=df
        )
        
        # 3. Publish Hour Heatmap
        hour_day_heatmap = df.pivot_table(
            index='publish_day', 
            columns='publish_hour',
            values='view_count', 
            aggfunc='mean'
        ).fillna(0)
        
        plots['publish_time_heatmap'] = create_plot(
            lambda **kwargs: sns.heatmap(
                kwargs['data'], 
                cmap='YlGnBu', 
                annot=False, 
                fmt='.0f'
            ),
            data=hour_day_heatmap
        )
        
        # 4. Duration Analysis
        plots['duration_analysis'] = create_plot(
            lambda **kwargs: sns.barplot(
                x=kwargs['data'].groupby('duration_range')['engagement_score'].mean().index,
                y=kwargs['data'].groupby('duration_range')['engagement_score'].mean().values,
                palette='rocket'
            ),
            data=df
        )
        
        # 5. Tag Count vs. Views
        tag_bins = [0, 5, 10, 15, 20, 25, 30, 100]
        tag_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '30+']
        df['tag_bin'] = pd.cut(df['tag_count'], bins=tag_bins, labels=tag_labels)
        
        plots['tags_analysis'] = create_plot(
            lambda **kwargs: sns.barplot(
                x=kwargs['data'].groupby('tag_bin')['view_count'].mean().index,
                y=kwargs['data'].groupby('tag_bin')['view_count'].mean().values,
                palette='crest'
            ),
            data=df
        )
        
        # 6. Top Channels
        top_channels_df = df.groupby('channel_title')['view_count'].mean().nlargest(10).reset_index()
        
        plots['top_channels'] = create_plot(
            lambda **kwargs: sns.barplot(
                y=kwargs['data']['channel_title'],
                x=kwargs['data']['view_count'],
                palette='mako'
            ),
            data=top_channels_df
        )
        
        return jsonify({
            'insights': insights,
            'plots': plots,
            'country': list(COUNTRIES.keys())[list(COUNTRIES.values()).index(country_code)]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/analyze-channel', methods=['POST'])
def analyze_channel():
    data = request.get_json()
    channel_url = data.get('channel_url', '')
    
    try:
        api_key = get_api_key()
        
        # Extract channel ID from URL
        channel_identifier = extract_channel_id(channel_url)
        if not channel_identifier:
            return jsonify({'error': 'Invalid YouTube channel URL. Please provide a valid channel URL.'})
        
        # Check cache
        cache_key = f"channel_{channel_identifier}"
        if cache_key in cache['channel_data']:
            return cache['channel_data'][cache_key]
        
        # Get channel info
        channel_info = get_channel_info(api_key, channel_identifier)
        if not channel_info:
            return jsonify({'error': 'Channel not found. Please check the URL and try again.'})
        
        # Get channel videos
        videos = get_channel_videos(api_key, channel_info['uploads_playlist'], max_results=50)
        if not videos:
            return jsonify({'error': 'No videos found for this channel.'})
        
        # Get category mapping
        category_mapping = get_category_mapping(api_key)
        
        # Process data
        df = process_data(videos, category_mapping)
        
        # Get trending videos for comparison
        trending_videos = get_trending_videos(api_key, 'US', max_results=50)
        trending_df = process_data(trending_videos, category_mapping)
        
        # Generate insights
        insights = generate_channel_insights(df, channel_info)
        
        # Generate improvement tips
        improvement_tips = generate_improvement_tips(df, trending_df)
        
        # Create visualizations
        plots = {}
        
        # 1. Video Performance Over Time
        # Sort by published date
        time_df = df.sort_values('published_at')
        
        plots['video_performance'] = create_plot(
            lambda **kwargs: plt.plot(
                kwargs['data']['published_at'], 
                kwargs['data']['view_count'], 
                'o-', 
                color='#FF0000'
            ),
            data=time_df
        )
        
        # 2. Content Category Distribution
        if 'category_name' in df.columns and not df['category_name'].isna().all():
            plots['channel_categories'] = create_plot(
                lambda **kwargs: sns.countplot(
                    y=kwargs['data']['category_name'],
                    order=kwargs['data']['category_name'].value_counts().index,
                    palette='viridis'
                ),
                data=df
            )
        else:
            # Fallback if categories are missing
            plots['channel_categories'] = create_plot(
                lambda **kwargs: plt.text(
                    0.5, 0.5, 
                    "Category data not available for this channel", 
                    ha='center', va='center', 
                    fontsize=12
                )
            )
        
        # 3. Upload Time Analysis
        if 'publish_hour' in df.columns and 'publish_day' in df.columns:
            hour_day_heatmap = df.pivot_table(
                index='publish_day', 
                columns='publish_hour',
                values='view_count', 
                aggfunc='mean'
            ).fillna(0)
            
            plots['upload_times'] = create_plot(
                lambda **kwargs: sns.heatmap(
                    kwargs['data'], 
                    cmap='YlGnBu', 
                    annot=False, 
                    fmt='.0f'
                ),
                data=hour_day_heatmap
            )
        else:
            # Fallback
            plots['upload_times'] = create_plot(
                lambda **kwargs: plt.text(
                    0.5, 0.5, 
                    "Upload time data not available", 
                    ha='center', va='center', 
                    fontsize=12
                )
            )
        
        # 4. Video Duration vs. Engagement
        plots['duration_analysis'] = create_plot(
            lambda **kwargs: sns.scatterplot(
                x='duration_seconds',
                y='engagement_score',
                hue='duration_range',
                palette='rocket',
                data=kwargs['data']
            ),
            data=df
        )
        
        # 5. Engagement Metrics
        engagement_df = pd.DataFrame({
            'Metric': ['Views', 'Likes', 'Comments'],
            'Average': [
                df['view_count'].mean(),
                df['like_count'].mean(),
                df['comment_count'].mean()
            ]
        })
        
        plots['engagement_metrics'] = create_plot(
            lambda **kwargs: sns.barplot(
                x='Metric',
                y='Average',
                palette='mako',
                data=kwargs['data']
            ),
            data=engagement_df
        )
        
        # Prepare response
        response = {
            'channel_name': channel_info['title'],
            'insights': insights,
            'plots': plots,
            'improvement_tips': improvement_tips
        }
        
        # Cache the results
        cache['channel_data'][cache_key] = response
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)

