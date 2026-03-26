@echo off

echo Running metrics for twilio
set BASE_URL=https://www.twilio.com/docs
set DOCS_PROVIDER=twilio
set SO_TAG=twilio-api
python main.py --metrics

echo Running metrics for discord
set BASE_URL=https://support.discord.com/hc/en-us
set DOCS_PROVIDER=discord
set SO_TAG=discord
python main.py --metrics

echo Running metrics for slack
set BASE_URL=https://slack.com/intl/en-in/help
set DOCS_PROVIDER=slack
set SO_TAG=slack-api
python main.py --metrics

echo Running metrics for amazons3
set BASE_URL=https://docs.aws.amazon.com/AmazonS3/latest/API/Welcome.html
set DOCS_PROVIDER=amazons3
set SO_TAG=amazon-s3
python main.py --metrics

echo Running metrics for notion
set BASE_URL=https://www.notion.com/help
set DOCS_PROVIDER=notion
set SO_TAG=notion-api
python main.py --metrics

echo Running metrics for stripe
set BASE_URL=https://docs.stripe.com/
set DOCS_PROVIDER=stripe
set SO_TAG=stripe-payments
python main.py --metrics

echo Done.