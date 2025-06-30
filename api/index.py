def handler(event, context):
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/html'},
        'body': '<h1>Hello from Python!</h1><p>Basic Python function working on Vercel.</p>'
    } 