def handler(event, context):
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'text/html',
        },
        'body': '''
        <html>
        <head><title>Text Complexity Estimator</title></head>
        <body>
            <h1>SUCCESS! Text Complexity Estimator is Working!</h1>
            <p>The site is now live and functional.</p>
            <p>This is the minimal working version.</p>
        </body>
        </html>
        '''
    } 